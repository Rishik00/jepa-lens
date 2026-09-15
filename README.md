# I-JEPA raw layer capture

The project now has two stages.

## 1. Capture

Choose an experiment YAML, edit its paths if needed, then run:

```bash
python main.py --config configs/mask_sizes.yaml
```

The target encoder is always read at its final layer. Predictor captures are
post-block. Each enabled condition is written beneath one timestamped suite:

```text
outputs/ijepa_experiments/<experiment-name>/<run-id>/<condition-name>/
```

Every YAML contains only the related conditions intended for that run:

| YAML | Conditions |
|---|---|
| `mask_sizes.yaml` | Clean 4x4, 5x5, 6x6, and 7x7 square masks |
| `strip_geometries.yaml` | Small and large horizontal/vertical strips |
| `object_masks.yaml` | Small and large object-aligned/random masks |
| `augmentations.yaml` | Clean, dimmed, noisy, and half-swapped inputs |
| `vit_baseline.yaml` | Clean I-JEPA capture plus the DINOv2 ViT baseline |

The YAML is the only source of runtime settings. The scripts no longer contain
a settings dictionary or read model/data paths from environment variables.

Object-aligned masks require JSON manifest rows with `bbox: [xmin, ymin, xmax,
ymax]`, or the equivalent four named fields. Coordinates are in the original
image and are transformed through the same resize and center crop as the image.

The optional `vit_baseline` uses a DINOv2 patch-14 ViT, giving the same 16x16
grid at 224-pixel input. It can download from Torch Hub or load weights from a
configured local checkpoint. Its selected patch tokens are saved alongside the
I-JEPA representations and automatically receive independent probes.

Each batch is written immediately as a shard, so the complete representation
dataset does not need to remain in CPU memory.

### Representation shards

Every file in `representation_shards/` contains:

| Key | Contents |
|---|---|
| `sample_ids` | Image IDs for this shard |
| `context_tokens` | Visible context-encoder patch representations |
| `target_patch_ids` | Actual target patch IDs for every image, including dynamic masks |
| `target_tokens` | Final target-encoder representations at the selected target patches |
| `predictor_hidden` | Raw internal target-token states from every selected predictor layer |
| `predictor_projected` | The same predictor states after the predictor's native final norm and projection |
| `vit_tokens` | Optional ViT patch representations for every selected ViT layer |

### Metric shards

Every file in `metric_shards/` contains raw, unaggregated measurements:

| Key | Contents |
|---|---|
| `corresponding_cosine` | Predicted patch 0 against target patch 0, patch 1 against patch 1, and so on |
| `corresponding_l1` | Mean absolute coordinate difference for corresponding patches |
| `corresponding_l2` | Euclidean distance for corresponding patches |
| `all_pairs_cosine` | Every predicted patch against every target patch |
| `all_pairs_l1` | All-pairs mean absolute distance |
| `all_pairs_l2` | All-pairs Euclidean distance |
| `predictor_self_cosine` | Every predicted patch against every other predicted patch |
| `target_self_cosine` | Every final target patch against every other final target patch |
| `pooled_cosine` | Cosine after averaging the selected predicted and target tokens |

`predictor_self_cosine` is the direct source for asking whether predicted
patches become more similar to one another or more distinct with depth.

No means or standard deviations across images are saved by the capture stage.
They can be calculated later from these raw values.

## 2. Analysis

Use the same YAML to analyze its newest capture:

```bash
python analyze.py --config configs/mask_sizes.yaml
```

For predictor layer 5, for example, the analysis averages the selected
projected patch tokens separately for every image. Within each fold it fits a
probe on the other folds and evaluates that probe on the held-out fold.

That process is repeated independently at every predictor layer. Separate
probes are fit for the final target encoder and every optional ViT layer. Probe
weights are not shared between representation sources.

By default, probing uses five stratified outer folds. Every image gets exactly
one out-of-fold prediction. Ridge strength is selected on a validation split
inside each outer training fold, after which the probe is refit on that full
outer training fold. A label-shuffled control follows the same procedure.

With `analysis.capture_dir: null`, the newest run belonging to that YAML's
`experiment_name` is selected. Set `capture_dir` in the YAML to a particular
run or condition directory to reproduce an older analysis. For a suite, every
condition is analyzed and combined comparison tables are written to its
suite-level `analysis/` directory.

### Analysis outputs

| File | Contents |
|---|---|
| `probe_layer_metrics.csv` | Out-of-fold accuracy, top-5, macro precision/recall/F1, and weighted F1 |
| `probe_predictions.csv` | One held-out prediction per image, including fold, scores, correctness, and patch dispersion |
| `probe_class_metrics.csv` | TP, FP, FN, support, precision, recall, and F1 for every class at every layer |
| `probe_confusions.csv` | Raw true-class/predicted-class confusion counts for every representation source |
| `probe_variance_subset_metrics.csv` | Probe accuracy split into low- vs high-variance test subsets at every layer |
| `probe_raw_outputs.pt` | Per-fold probe weights, fold IDs, and all out-of-fold class scores |
| `probe_selected_ridges.csv` | Ridge strength selected independently in every outer fold |
| `patch_target_ranks.csv` | For every predicted patch: the corresponding target's rank, best target, and cosine margin |
| `patch_alignment_class_metrics.csv` | Top-1/top-2/top-4 target-position matching broken down by image class and predictor layer |
| `patch_geometry_by_image.csv` | Predicted-versus-target within-block patch similarity for every image and layer |
| `patch_geometry_class_metrics.csv` | Within-block patch similarity broken down by image class and layer |
| `pooled_distribution_stats.csv` | Population spread of the mean-pooled vectors per layer: mean/std norm, mean per-dim std, total variance |
| `pooled_distribution_raw.pt` | Raw per-dimension mean/std vectors for every representation |
| `corresponding_patch_metrics.csv` | Readable CSV export of raw corresponding-patch cosine, L1, and L2 values |
| `pooled_cosine_by_image.csv` | One mean-pooled cosine per image and predictor layer |

Suite-level outputs include combined probe tables, raw per-image pooled cosine
values across conditions, and paired pooled-cosine changes relative to the
configured clean reference condition.

**Patch dispersion** is how far an image's individual patch tokens sit from
their own pooled mean, measured per image and per representation as mean
Euclidean distance and mean cosine distance from the centroid. Each test image
carries its dispersion in `probe_predictions.csv`, and
`probe_variance_subset_metrics.csv` splits the test set at the
`variance_quantile` (default median) of each representation's L2 dispersion so
you can see whether high-spread images are harder for the probe.

The dense all-pairs and within-predictor matrices remain in the metric shards;
expanding them into CSV would produce millions of rows.

Mahalanobis distance and interventions are intentionally not included yet.
Both can be added as analysis steps over the saved representations without
running I-JEPA again.
