# I-JEPA interpretability experiment roadmap

This file records the experiment directions proposed in our discussion. It is
also the implementation checklist for `main.py` and `analyze.py`.

## Implementation status

- Target-mask size sweep: implemented as selectable 4x4, 5x5, 6x6, and 7x7
  capture conditions.
- Stronger probes: implemented with stratified out-of-fold predictions, inner
  ridge selection, shuffled labels, per-class metrics, confusions, and logits.
- ViT baseline: implemented as an optional same-grid DINOv2 patch-14 capture
  with one independent probe per selected layer.
- Mask geometries: implemented for two equal-area strip pairs and for
  annotation-centered versus deterministic random masks.
- Image perturbations: dimming, Gaussian pixel noise, and left/right half swap
  are implemented with either context-only or both-branch scope.

Experiment groups are defined in separate canonical YAMLs under `configs/`.
Running one YAML creates a fresh run directory under that YAML's
`experiment_name`; no source-code settings need to be edited.

## Current measurement convention

- The model receives a 224 by 224 image divided into a 16 by 16 grid of model
  patches (256 positions total).
- A target mask selects positions that the predictor must predict from the
  remaining visible context.
- Predictor captures are currently **post-block**: `predictor_00` is the state
  after predictor block 0 has completed its attention, MLP, and residual
  updates. The input to block 0 is not currently saved.
- At each captured predictor block, its target-position states pass through the
  predictor's shared final normalization and projection before comparison with
  the final target-encoder states at the same positions.

## 1. Target-mask size sweep

Question: how does the predictor trajectory change when it must predict more
image positions?

Keep the model's physical patch size fixed at 14 by 14 pixels and vary the
number of target positions. A useful square-mask sweep is:

| Mask | Target tokens | Fraction of the 256-token grid |
|---|---:|---:|
| 4 by 4 | 16 | 6.25% |
| 5 by 5 | 25 | 9.77% |
| 6 by 6 | 36 | 14.06% |
| 7 by 7 | 49 | 19.14% |

The final level is close to the 15--20% target-block scale used to train the
original I-JEPA model. For each size, retain raw per-image, per-layer, and
per-target-position distances plus the independently trained probe outputs.

Changing the model's actual 14-pixel patch resolution is a separate experiment
requiring another architecture/checkpoint; it should not be conflated with
increasing the number of predicted patches.

## 2. Strengthen semantic probes and remove retrieval

Question: which class information is linearly recoverable from each predictor
layer, and exactly where does it fail?

Retrieval is removed from `analyze.py`. The probe analysis should focus on:

- one independent probe per predictor layer;
- an independent probe on final target-encoder representations;
- identical data splits and fitting procedure for every representation source;
- per-class precision, recall, F1, support, confusion counts, and raw logits;
- macro and weighted summaries rather than accuracy alone;
- shuffled-label controls; and
- out-of-fold predictions when scaling to 20 images per class, so every sample
  can contribute to a held-out prediction instead of leaving only four test
  images per class in one 80/20 split.

## 3. ViT representation baseline

Question: how linearly readable are the same spatial regions in an ordinary
ViT compared with the I-JEPA predictor and target encoder?

For the same images and selected spatial positions:

1. Extract the ViT patch tokens at those positions from each chosen ViT layer.
2. Pool only those selected tokens per image, exactly as for I-JEPA.
3. Train an independent probe at each ViT layer using the same folds, classes,
   metrics, and regularization selection.

A full-image ViT is a representation-quality reference, not a completion
baseline: it can directly see pixels inside the selected target region, while
the I-JEPA predictor cannot. A masked-input ViT can be reported separately,
but a standard discriminative ViT was not trained to infer mask tokens. Model
patch grids must be aligned before making position-by-position comparisons.

## 4. Mask geometry at two target-size levels

Question: does prediction depend on mask orientation, object alignment, or
only the number of missing positions?

Use equal-area masks so geometry is not confounded with target-token count.
Two concrete levels on the 16 by 16 grid are:

| Level | Horizontal strip | Vertical strip | Target tokens |
|---|---|---|---:|
| Small | 2 by 8 | 8 by 2 | 16 |
| Large | 4 by 12 | 12 by 4 | 48 |

For object alignment, compare an object-aligned mask with a random-position
mask matched for token count, shape, and image. Object alignment requires
localization boxes or segmentation annotations; ImageNet classification folder
labels alone do not define an object mask.

Report paired differences for the same images rather than comparing unrelated
sets of images. Also record each mask's exact patch IDs and its overlap with the
object annotation.

## 5. Image perturbations

Question: does the layerwise trajectory reflect robustness, denoising, global
layout, or local visual content?

Planned perturbations:

- controlled dimming;
- controlled additive noise; and
- split an image into two halves and reassemble it, with the exact rearrangement
  recorded (for example, swapping left and right halves).

There are two distinct protocols:

- **Corrupted context, clean target:** the context encoder receives the
  perturbed image while the target representation comes from the clean image.
  This directly tests recovery toward a clean latent target.
- **Both branches corrupted:** context and target encoders receive the same
  perturbed image. This tests whether the normal prediction relationship is
  preserved under the perturbation.

Run paired clean and perturbed versions using identical image IDs and masks,
then save per-image and per-patch changes rather than only aggregate means.

## 6. Further distance and circuit-style analyses

Reserved for concrete hypotheses after reviewing the circuits literature.
Candidate metrics or interventions should be added only when each answers a
specific question beyond the existing cosine, L1, L2, self-similarity, probe,
and target-position-ranking measurements.

## Open implementation choices

- Which ViT checkpoint is the fairest baseline while preserving a compatible
  patch grid?
- Which localization dataset or annotations will define object-aligned masks?
- Whether the first perturbation run should use clean targets, matched
  perturbed targets, or report both as separate conditions.
