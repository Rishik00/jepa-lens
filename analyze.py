"""Analyze a raw capture produced by main.py.

This script does not run I-JEPA. It loads saved representations and performs:

1. one independently trained semantic probe per predictor layer;
2. one independently trained probe on final target-encoder representations;
3. patch-position alignment analysis; and
4. readable CSV export of raw corresponding-patch and pooled cosine values.

Run ``python analyze.py --config configs/<experiment>.yaml``. Analyses can be
rerun many times on one capture without rerunning the large vision model.
"""

from __future__ import annotations

import csv
import json
import re
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
import torch.nn.functional as F

from config_utils import load_project_config

try:
    from torchvision.models import ResNet50_Weights

    IMAGENET_CLASS_NAMES: Sequence[str] | None = ResNet50_Weights.DEFAULT.meta["categories"]
except (ImportError, AttributeError, KeyError):
    IMAGENET_CLASS_NAMES = None


ANALYSIS_CONFIG: dict[str, Any] = {}


def resolve_capture_dirs() -> list[Path]:
    configured = ANALYSIS_CONFIG["capture_dir"]
    if configured is not None:
        path = Path(configured)
    else:
        root = Path(ANALYSIS_CONFIG["capture_root"])
        candidates = sorted(path for path in root.iterdir() if path.is_dir())
        if not candidates:
            raise FileNotFoundError(f"No capture directories found inside {root}.")
        path = candidates[-1]
    if (path / "capture_metadata.json").is_file():
        return [path]
    conditions = sorted(
        child for child in path.iterdir()
        if child.is_dir() and (child / "capture_metadata.json").is_file()
    )
    if not conditions:
        raise FileNotFoundError(f"Not a raw capture directory or suite: {path}")
    return conditions


def read_manifest(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    rows.sort(key=lambda row: int(row["sample_id"]))
    return rows


def normalize_rows(array: np.ndarray) -> np.ndarray:
    array = np.asarray(array, dtype=np.float32)
    return array / np.clip(np.linalg.norm(array, axis=-1, keepdims=True), 1e-12, None)


def fit_ridge(features: np.ndarray, labels: np.ndarray, n_classes: int, ridge: float) -> np.ndarray:
    x = normalize_rows(features).astype(np.float64)
    y = np.eye(n_classes, dtype=np.float64)[labels]
    if x.shape[0] <= x.shape[1]:
        weights = x.T @ np.linalg.solve(x @ x.T + ridge * np.eye(x.shape[0]), y)
    else:
        weights = np.linalg.solve(x.T @ x + ridge * np.eye(x.shape[1]), x.T @ y)
    return weights.astype(np.float32)


def topk_metrics(logits: np.ndarray, labels: np.ndarray, k: int = 5) -> tuple[float, float, np.ndarray]:
    ranking = np.argsort(logits, axis=1)[:, ::-1]
    top1 = float(np.mean(ranking[:, 0] == labels))
    topk = float(np.mean([label in row[: min(k, row.size)] for label, row in zip(labels, ranking)]))
    return top1, topk, ranking


def imagenet_display_name(raw_name: str) -> str:
    """Expand labels such as 220 or class_0220 when metadata is available."""
    match = re.fullmatch(r"(?:class_)?(\d{1,4})", raw_name)
    if match is None or IMAGENET_CLASS_NAMES is None:
        return raw_name
    index = int(match.group(1))
    if not 0 <= index < len(IMAGENET_CLASS_NAMES):
        return raw_name
    return str(IMAGENET_CLASS_NAMES[index])


def per_class_statistics(
    true_labels: np.ndarray, predicted_labels: np.ndarray, n_classes: int
) -> list[dict[str, float | int]]:
    """Calculate one-vs-rest counts and scores for every class."""
    rows: list[dict[str, float | int]] = []
    for class_id in range(n_classes):
        tp = int(np.sum((true_labels == class_id) & (predicted_labels == class_id)))
        fp = int(np.sum((true_labels != class_id) & (predicted_labels == class_id)))
        fn = int(np.sum((true_labels == class_id) & (predicted_labels != class_id)))
        support = int(np.sum(true_labels == class_id))
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
        rows.append(
            {
                "true_positive": tp,
                "false_positive": fp,
                "false_negative": fn,
                "support": support,
                "precision": precision,
                "recall": recall,
                "f1": f1,
            }
        )
    return rows


def patch_dispersion(patches: torch.Tensor, pooled: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Spread of individual patch tokens around their own pooled mean.

    ``patches`` has a trailing (patch, feature) pair and ``pooled`` is the same
    tensor averaged over the patch axis. Returns two per-image (or per-image,
    per-layer) scalars: mean Euclidean distance from the centroid, and mean
    cosine distance (1 - cosine) from the centroid.
    """
    centroid = pooled.unsqueeze(-2)
    l2 = (patches - centroid).norm(dim=-1).mean(dim=-1)
    cosine_distance = (1.0 - F.cosine_similarity(patches, centroid, dim=-1)).mean(dim=-1)
    return l2, cosine_distance


class PooledData:
    """Pooled vectors and per-image patch dispersion for every representation."""

    def __init__(
        self,
        sample_ids: np.ndarray,
        target: np.ndarray,
        predictor: np.ndarray,
        dispersion_target_l2: np.ndarray,
        dispersion_target_cos: np.ndarray,
        dispersion_predictor_l2: np.ndarray,
        dispersion_predictor_cos: np.ndarray,
        vit: np.ndarray | None,
        dispersion_vit_l2: np.ndarray | None,
        dispersion_vit_cos: np.ndarray | None,
    ) -> None:
        self.sample_ids = sample_ids
        self.target = target
        self.predictor = predictor
        self.dispersion_target_l2 = dispersion_target_l2
        self.dispersion_target_cos = dispersion_target_cos
        self.dispersion_predictor_l2 = dispersion_predictor_l2
        self.dispersion_predictor_cos = dispersion_predictor_cos
        self.vit = vit
        self.dispersion_vit_l2 = dispersion_vit_l2
        self.dispersion_vit_cos = dispersion_vit_cos


def load_pooled_representations(capture_dir: Path) -> PooledData:
    sample_parts: list[torch.Tensor] = []
    target_parts: list[torch.Tensor] = []
    predictor_parts: list[torch.Tensor] = []
    target_l2_parts: list[torch.Tensor] = []
    target_cos_parts: list[torch.Tensor] = []
    predictor_l2_parts: list[torch.Tensor] = []
    predictor_cos_parts: list[torch.Tensor] = []
    vit_parts: list[torch.Tensor] = []
    vit_l2_parts: list[torch.Tensor] = []
    vit_cos_parts: list[torch.Tensor] = []
    for path in sorted((capture_dir / "representation_shards").glob("shard_*.pt")):
        shard = torch.load(path, map_location="cpu", weights_only=True)
        target_tokens = shard["target_tokens"].float()
        predictor_tokens = shard["predictor_projected"].float()
        target_pooled = target_tokens.mean(dim=1)
        predictor_pooled = predictor_tokens.mean(dim=2)
        sample_parts.append(shard["sample_ids"])
        target_parts.append(target_pooled)
        predictor_parts.append(predictor_pooled)
        target_l2, target_cos = patch_dispersion(target_tokens, target_pooled)
        predictor_l2, predictor_cos = patch_dispersion(predictor_tokens, predictor_pooled)
        target_l2_parts.append(target_l2)
        target_cos_parts.append(target_cos)
        predictor_l2_parts.append(predictor_l2)
        predictor_cos_parts.append(predictor_cos)
        if "vit_tokens" in shard:
            vit_tokens = shard["vit_tokens"].float()
            vit_pooled = vit_tokens.mean(dim=2)
            vit_l2, vit_cos = patch_dispersion(vit_tokens, vit_pooled)
            vit_parts.append(vit_pooled)
            vit_l2_parts.append(vit_l2)
            vit_cos_parts.append(vit_cos)
    if not sample_parts:
        raise FileNotFoundError("No representation shards found.")
    sample_ids = torch.cat(sample_parts).numpy()
    order = np.argsort(sample_ids)
    return PooledData(
        sample_ids=sample_ids[order],
        target=torch.cat(target_parts).numpy()[order],
        predictor=torch.cat(predictor_parts).numpy()[order],
        dispersion_target_l2=torch.cat(target_l2_parts).numpy()[order],
        dispersion_target_cos=torch.cat(target_cos_parts).numpy()[order],
        dispersion_predictor_l2=torch.cat(predictor_l2_parts).numpy()[order],
        dispersion_predictor_cos=torch.cat(predictor_cos_parts).numpy()[order],
        vit=torch.cat(vit_parts).numpy()[order] if vit_parts else None,
        dispersion_vit_l2=torch.cat(vit_l2_parts).numpy()[order] if vit_l2_parts else None,
        dispersion_vit_cos=torch.cat(vit_cos_parts).numpy()[order] if vit_cos_parts else None,
    )


def make_feature_sets(
    data: PooledData, predictor_layers: Sequence[int], vit_layers: Sequence[int]
) -> list[tuple[str, np.ndarray, np.ndarray, np.ndarray]]:
    feature_sets = [
        (
            f"predictor_{layer:02d}",
            data.predictor[:, index],
            data.dispersion_predictor_l2[:, index],
            data.dispersion_predictor_cos[:, index],
        )
        for index, layer in enumerate(predictor_layers)
    ]
    feature_sets.append(
        ("target_final", data.target, data.dispersion_target_l2, data.dispersion_target_cos)
    )
    if data.vit is not None:
        if data.dispersion_vit_l2 is None or data.dispersion_vit_cos is None:
            raise RuntimeError("Incomplete ViT dispersion data.")
        feature_sets.extend(
            (
                f"vit_{layer:02d}",
                data.vit[:, index],
                data.dispersion_vit_l2[:, index],
                data.dispersion_vit_cos[:, index],
            )
            for index, layer in enumerate(vit_layers)
        )
    return feature_sets


def write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def run_probes(
    output_dir: Path,
    manifest: Sequence[dict[str, str]],
    data: PooledData,
    predictor_layers: Sequence[int],
    vit_layers: Sequence[int],
) -> None:
    labels = np.asarray([int(row["label_id"]) for row in manifest], dtype=np.int64)
    names_by_id = {int(row["label_id"]): row["class_name"] for row in manifest}
    class_names = [names_by_id[index] for index in range(len(names_by_id))]
    display_names = [imagenet_display_name(name) for name in class_names]
    n_classes = len(class_names)
    train = np.asarray([i for i, row in enumerate(manifest) if row["split"] == "train"])
    test = np.asarray([i for i, row in enumerate(manifest) if row["split"] == "test"])
    quantile = float(ANALYSIS_CONFIG["variance_quantile"])

    # (name, pooled features, per-image L2 dispersion, per-image cosine dispersion)
    feature_sets = make_feature_sets(data, predictor_layers, vit_layers)

    metric_rows: list[dict[str, Any]] = []
    prediction_rows: list[dict[str, Any]] = []
    class_rows: list[dict[str, Any]] = []
    confusion_rows: list[dict[str, Any]] = []
    variance_rows: list[dict[str, Any]] = []
    saved_weights: dict[str, torch.Tensor] = {}
    saved_test_logits: dict[str, torch.Tensor] = {}

    for representation_name, features, dispersion_l2, dispersion_cos in feature_sets:
        weights = fit_ridge(
            features[train], labels[train], n_classes, float(ANALYSIS_CONFIG["ridge"])
        )
        train_logits = normalize_rows(features[train]) @ weights
        test_logits = normalize_rows(features[test]) @ weights
        train_accuracy, train_top5, _ = topk_metrics(train_logits, labels[train])
        test_accuracy, test_top5, ranking = topk_metrics(test_logits, labels[test])
        predicted_test_labels = ranking[:, 0]
        class_statistics = per_class_statistics(labels[test], predicted_test_labels, n_classes)
        macro_precision = float(np.mean([row["precision"] for row in class_statistics]))
        macro_recall = float(np.mean([row["recall"] for row in class_statistics]))
        macro_f1 = float(np.mean([row["f1"] for row in class_statistics]))
        total_support = sum(int(row["support"]) for row in class_statistics)
        weighted_f1 = float(
            sum(float(row["f1"]) * int(row["support"]) for row in class_statistics)
            / total_support
        )
        metric_rows.append(
            {
                "representation": representation_name,
                "train_accuracy": train_accuracy,
                "train_top5_accuracy": train_top5,
                "test_accuracy": test_accuracy,
                "test_top5_accuracy": test_top5,
                "macro_precision": macro_precision,
                "macro_recall": macro_recall,
                "macro_f1": macro_f1,
                "weighted_f1": weighted_f1,
                "n_train": len(train),
                "n_test": len(test),
            }
        )
        saved_weights[representation_name] = torch.from_numpy(weights)
        saved_test_logits[representation_name] = torch.from_numpy(test_logits)

        test_correct = (ranking[:, 0] == labels[test]).astype(np.int64)
        for query_index, sample_index in enumerate(test):
            ranked = ranking[query_index, : min(5, n_classes)]
            prediction_rows.append(
                {
                    "representation": representation_name,
                    "sample_id": int(sample_index),
                    "path": manifest[sample_index]["path"],
                    "true_class": manifest[sample_index]["class_name"],
                    "true_class_name": display_names[int(labels[sample_index])],
                    "predicted_class": class_names[int(ranked[0])],
                    "predicted_class_name": display_names[int(ranked[0])],
                    "correct": int(test_correct[query_index]),
                    "top1_score": float(test_logits[query_index, ranked[0]]),
                    "top1_margin": float(test_logits[query_index, ranked[0]] - test_logits[query_index, ranked[1]]),
                    "dispersion_l2": float(dispersion_l2[sample_index]),
                    "dispersion_centroid_cosine": float(dispersion_cos[sample_index]),
                    "top5_classes": json.dumps([class_names[int(index)] for index in ranked]),
                    "top5_class_names": json.dumps([display_names[int(index)] for index in ranked]),
                    "top5_scores": json.dumps([float(test_logits[query_index, index]) for index in ranked]),
                }
            )

        for class_id, class_name in enumerate(class_names):
            stats = class_statistics[class_id]
            class_rows.append(
                {
                    "representation": representation_name,
                    "class_id": class_id,
                    "class_name": class_name,
                    "imagenet_name": display_names[class_id],
                    "correct": stats["true_positive"],
                    "accuracy": stats["recall"],
                    **stats,
                }
            )

        confusion_counts: dict[tuple[int, int], int] = {}
        for true_class, predicted_class in zip(labels[test], predicted_test_labels):
            key = (int(true_class), int(predicted_class))
            confusion_counts[key] = confusion_counts.get(key, 0) + 1
        for (true_class, predicted_class), count in sorted(confusion_counts.items()):
            confusion_rows.append(
                {
                    "representation": representation_name,
                    "true_class": class_names[true_class],
                    "true_class_name": display_names[true_class],
                    "predicted_class": class_names[predicted_class],
                    "predicted_class_name": display_names[predicted_class],
                    "count": count,
                    "correct": int(true_class == predicted_class),
                }
            )

        # Split the test set by how far this representation's patches spread
        # around their pooled mean, then compare probe accuracy across subsets.
        test_dispersion = dispersion_l2[test]
        threshold = float(np.quantile(test_dispersion, quantile))
        high_mask = test_dispersion >= threshold
        for subset_name, subset_mask in (("low_variance", ~high_mask), ("high_variance", high_mask)):
            support = int(subset_mask.sum())
            variance_rows.append(
                {
                    "representation": representation_name,
                    "subset": subset_name,
                    "dispersion_quantile": quantile,
                    "dispersion_threshold_l2": threshold,
                    "support": support,
                    "accuracy": float(test_correct[subset_mask].mean()) if support else "",
                    "mean_dispersion_l2": float(test_dispersion[subset_mask].mean()) if support else "",
                }
            )

    write_csv(output_dir / "probe_layer_metrics.csv", metric_rows)
    write_csv(output_dir / "probe_predictions.csv", prediction_rows)
    write_csv(output_dir / "probe_class_metrics.csv", class_rows)
    write_csv(output_dir / "probe_confusions.csv", confusion_rows)
    write_csv(output_dir / "probe_variance_subset_metrics.csv", variance_rows)
    torch.save(
        {"weights": saved_weights, "test_logits": saved_test_logits},
        output_dir / "probe_raw_outputs.pt",
    )


def stratified_fold_ids(labels: np.ndarray, n_folds: int, seed: int) -> np.ndarray:
    counts = np.bincount(labels)
    if counts.min() < n_folds:
        raise ValueError(
            f"Every class needs at least {n_folds} samples for {n_folds}-fold CV; "
            f"smallest class has {counts.min()}."
        )
    rng = np.random.default_rng(seed)
    fold_ids = np.empty(len(labels), dtype=np.int64)
    for class_id in range(len(counts)):
        indices = np.flatnonzero(labels == class_id)
        rng.shuffle(indices)
        fold_ids[indices] = np.arange(len(indices)) % n_folds
    return fold_ids


def choose_ridge(
    features: np.ndarray,
    fitting_labels: np.ndarray,
    indices: np.ndarray,
    n_classes: int,
    ridge_values: Sequence[float],
    seed: int,
) -> float:
    """Choose ridge on a stratified validation subset inside an outer fold."""
    relative_train, relative_validation = _stratified_holdout(
        fitting_labels[indices], 0.2, seed
    )
    train = indices[relative_train]
    validation = indices[relative_validation]
    best_ridge = float(ridge_values[0])
    best_macro_f1 = -1.0
    for ridge in ridge_values:
        weights = fit_ridge(features[train], fitting_labels[train], n_classes, float(ridge))
        logits = normalize_rows(features[validation]) @ weights
        predicted = np.argmax(logits, axis=1)
        stats = per_class_statistics(fitting_labels[validation], predicted, n_classes)
        macro_f1 = float(np.mean([row["f1"] for row in stats]))
        if macro_f1 > best_macro_f1:
            best_macro_f1 = macro_f1
            best_ridge = float(ridge)
    return best_ridge


def _stratified_holdout(
    labels: np.ndarray, test_fraction: float, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    train: list[int] = []
    test: list[int] = []
    for label in np.unique(labels):
        indices = np.flatnonzero(labels == label)
        rng.shuffle(indices)
        if len(indices) < 2:
            train.extend(indices.tolist())
            continue
        n_test = min(max(1, int(round(len(indices) * test_fraction))), len(indices) - 1)
        test.extend(indices[:n_test].tolist())
        train.extend(indices[n_test:].tolist())
    if not test:
        raise ValueError("Could not form an inner validation split.")
    return np.asarray(train, dtype=np.int64), np.asarray(test, dtype=np.int64)


def run_cross_validated_probes(
    output_dir: Path,
    manifest: Sequence[dict[str, str]],
    data: PooledData,
    predictor_layers: Sequence[int],
    vit_layers: Sequence[int],
) -> None:
    """Create one held-out prediction per image for every representation source."""
    labels = np.asarray([int(row["label_id"]) for row in manifest], dtype=np.int64)
    names_by_id = {int(row["label_id"]): row["class_name"] for row in manifest}
    class_names = [names_by_id[index] for index in range(len(names_by_id))]
    display_names = [imagenet_display_name(name) for name in class_names]
    n_classes = len(class_names)
    n_folds = int(ANALYSIS_CONFIG["n_folds"])
    seed = int(ANALYSIS_CONFIG["probe_seed"])
    ridge_values = [float(value) for value in ANALYSIS_CONFIG["ridge_values"]]
    if not ridge_values or any(value <= 0 for value in ridge_values):
        raise ValueError("ridge_values must contain positive values.")
    fold_ids = stratified_fold_ids(labels, n_folds, seed)
    quantile = float(ANALYSIS_CONFIG["variance_quantile"])
    feature_sets = make_feature_sets(data, predictor_layers, vit_layers)

    metric_rows: list[dict[str, Any]] = []
    prediction_rows: list[dict[str, Any]] = []
    class_rows: list[dict[str, Any]] = []
    confusion_rows: list[dict[str, Any]] = []
    variance_rows: list[dict[str, Any]] = []
    ridge_rows: list[dict[str, Any]] = []
    saved_logits: dict[str, torch.Tensor] = {}
    saved_weights: dict[str, torch.Tensor] = {}

    for representation_index, (representation_name, features, dispersion_l2, dispersion_cos) in enumerate(feature_sets):
        controls = ["real"]
        if ANALYSIS_CONFIG["run_shuffled_label_control"]:
            controls.append("shuffled")
        for control in controls:
            fitting_labels = labels.copy()
            if control == "shuffled":
                rng = np.random.default_rng(seed + 100_003 + representation_index)
                fitting_labels = rng.permutation(fitting_labels)
            oof_logits = np.full((len(labels), n_classes), np.nan, dtype=np.float32)

            for fold in range(n_folds):
                test = np.flatnonzero(fold_ids == fold)
                train = np.flatnonzero(fold_ids != fold)
                ridge = choose_ridge(
                    features,
                    fitting_labels,
                    train,
                    n_classes,
                    ridge_values,
                    seed + fold,
                )
                weights = fit_ridge(features[train], fitting_labels[train], n_classes, ridge)
                oof_logits[test] = normalize_rows(features[test]) @ weights
                saved_weights[f"{representation_name}__{control}__fold_{fold}"] = torch.from_numpy(weights)
                ridge_rows.append(
                    {
                        "representation": representation_name,
                        "control": control,
                        "fold": fold,
                        "ridge": ridge,
                        "n_train": len(train),
                        "n_test": len(test),
                    }
                )
            if np.isnan(oof_logits).any():
                raise RuntimeError("Cross-validation did not produce every out-of-fold score.")

            predicted = np.argmax(oof_logits, axis=1)
            ranking = np.argsort(oof_logits, axis=1)[:, ::-1]
            top1, top5, _ = topk_metrics(oof_logits, labels)
            stats = per_class_statistics(labels, predicted, n_classes)
            macro_precision = float(np.mean([row["precision"] for row in stats]))
            macro_recall = float(np.mean([row["recall"] for row in stats]))
            macro_f1 = float(np.mean([row["f1"] for row in stats]))
            weighted_f1 = float(
                sum(float(row["f1"]) * int(row["support"]) for row in stats) / len(labels)
            )
            metric_rows.append(
                {
                    "representation": representation_name,
                    "control": control,
                    "evaluation": f"{n_folds}_fold_out_of_fold",
                    "accuracy": top1,
                    "top5_accuracy": top5,
                    "macro_precision": macro_precision,
                    "macro_recall": macro_recall,
                    "macro_f1": macro_f1,
                    "weighted_f1": weighted_f1,
                    "n_samples": len(labels),
                }
            )
            saved_logits[f"{representation_name}__{control}"] = torch.from_numpy(oof_logits)

            for sample_index in range(len(labels)):
                ranked = ranking[sample_index, : min(5, n_classes)]
                prediction_rows.append(
                    {
                        "representation": representation_name,
                        "control": control,
                        "fold": int(fold_ids[sample_index]),
                        "sample_id": sample_index,
                        "path": manifest[sample_index]["path"],
                        "true_class": manifest[sample_index]["class_name"],
                        "true_class_name": display_names[int(labels[sample_index])],
                        "predicted_class": class_names[int(ranked[0])],
                        "predicted_class_name": display_names[int(ranked[0])],
                        "correct": int(ranked[0] == labels[sample_index]),
                        "top1_score": float(oof_logits[sample_index, ranked[0]]),
                        "top1_margin": float(oof_logits[sample_index, ranked[0]] - oof_logits[sample_index, ranked[1]]),
                        "dispersion_l2": float(dispersion_l2[sample_index]),
                        "dispersion_centroid_cosine": float(dispersion_cos[sample_index]),
                        "top5_classes": json.dumps([class_names[int(index)] for index in ranked]),
                        "top5_class_names": json.dumps([display_names[int(index)] for index in ranked]),
                        "top5_scores": json.dumps([float(oof_logits[sample_index, index]) for index in ranked]),
                    }
                )

            for class_id, class_name in enumerate(class_names):
                class_rows.append(
                    {
                        "representation": representation_name,
                        "control": control,
                        "class_id": class_id,
                        "class_name": class_name,
                        "imagenet_name": display_names[class_id],
                        **stats[class_id],
                    }
                )
            confusion_counts: dict[tuple[int, int], int] = {}
            for true_class, predicted_class in zip(labels, predicted):
                key = (int(true_class), int(predicted_class))
                confusion_counts[key] = confusion_counts.get(key, 0) + 1
            for (true_class, predicted_class), count in sorted(confusion_counts.items()):
                confusion_rows.append(
                    {
                        "representation": representation_name,
                        "control": control,
                        "true_class": class_names[true_class],
                        "true_class_name": display_names[true_class],
                        "predicted_class": class_names[predicted_class],
                        "predicted_class_name": display_names[predicted_class],
                        "count": count,
                        "correct": int(true_class == predicted_class),
                    }
                )

            threshold = float(np.quantile(dispersion_l2, quantile))
            high_mask = dispersion_l2 >= threshold
            correct = predicted == labels
            for subset_name, subset_mask in (("low_variance", ~high_mask), ("high_variance", high_mask)):
                support = int(subset_mask.sum())
                variance_rows.append(
                    {
                        "representation": representation_name,
                        "control": control,
                        "subset": subset_name,
                        "dispersion_quantile": quantile,
                        "dispersion_threshold_l2": threshold,
                        "support": support,
                        "accuracy": float(correct[subset_mask].mean()) if support else "",
                        "mean_dispersion_l2": float(dispersion_l2[subset_mask].mean()) if support else "",
                    }
                )

    write_csv(output_dir / "probe_layer_metrics.csv", metric_rows)
    write_csv(output_dir / "probe_predictions.csv", prediction_rows)
    write_csv(output_dir / "probe_class_metrics.csv", class_rows)
    write_csv(output_dir / "probe_confusions.csv", confusion_rows)
    write_csv(output_dir / "probe_variance_subset_metrics.csv", variance_rows)
    write_csv(output_dir / "probe_selected_ridges.csv", ridge_rows)
    torch.save(
        {"fold_ids": torch.from_numpy(fold_ids), "weights": saved_weights, "oof_logits": saved_logits},
        output_dir / "probe_raw_outputs.pt",
    )


def run_patch_alignment(
    capture_dir: Path,
    output_dir: Path,
    manifest: Sequence[dict[str, str]],
    predictor_layers: Sequence[int],
    target_patch_ids: Sequence[int] | None,
) -> None:
    """Record target-position ranks and within-image patch geometry by class."""
    patch_path = output_dir / "patch_target_ranks.csv"
    geometry_rows: list[dict[str, Any]] = []
    alignment_totals: dict[tuple[int, str], dict[str, float]] = {}
    geometry_totals: dict[tuple[int, str], dict[str, float]] = {}
    first_shard_path = next(iter(sorted((capture_dir / "metric_shards").glob("shard_*.pt"))), None)
    if first_shard_path is None:
        raise FileNotFoundError("No metric shards found.")
    first_shard = torch.load(first_shard_path, map_location="cpu", weights_only=True)
    n_patches = int(first_shard["all_pairs_cosine"].shape[-1])
    off_diagonal = ~torch.eye(n_patches, dtype=torch.bool)

    with patch_path.open("w", newline="", encoding="utf-8") as handle:
        fieldnames = (
            "sample_id", "class_name", "imagenet_name", "predictor_layer",
            "predicted_patch_offset", "predicted_patch_id", "best_target_offset",
            "best_target_patch_id", "corresponding_target_rank", "top1_correct",
            "top2_correct", "top4_correct", "corresponding_cosine",
            "best_incorrect_cosine", "corresponding_margin_over_best_incorrect",
        )
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for path in sorted((capture_dir / "metric_shards").glob("shard_*.pt")):
            shard = torch.load(path, map_location="cpu", weights_only=True)
            all_pairs = shard["all_pairs_cosine"]
            predictor_self = shard["predictor_self_cosine"]
            target_self = shard["target_self_cosine"]
            for batch_index, sample_id_tensor in enumerate(shard["sample_ids"]):
                sample_id = int(sample_id_tensor)
                image_patch_ids = (
                    [int(value) for value in shard["target_patch_ids"][batch_index]]
                    if "target_patch_ids" in shard
                    else list(target_patch_ids or ())
                )
                if len(image_patch_ids) != n_patches:
                    raise RuntimeError("Target patch IDs do not match the metric matrix.")
                class_name = manifest[sample_id]["class_name"]
                display_name = imagenet_display_name(class_name)
                target_self_mean = float(target_self[batch_index][off_diagonal].mean())
                for layer_offset, layer in enumerate(predictor_layers):
                    matrix = all_pairs[batch_index, layer_offset]
                    ranking = torch.argsort(matrix, dim=-1, descending=True)
                    truth = torch.arange(n_patches).unsqueeze(1)
                    correct_ranks = (ranking == truth).to(torch.int64).argmax(dim=-1) + 1
                    diagonal = matrix.diagonal()
                    without_diagonal = matrix.clone()
                    without_diagonal[torch.arange(n_patches), torch.arange(n_patches)] = -torch.inf
                    best_incorrect = without_diagonal.max(dim=-1).values
                    best_targets = matrix.argmax(dim=-1)
                    predictor_self_mean = float(
                        predictor_self[batch_index, layer_offset][off_diagonal].mean()
                    )
                    geometry_rows.append(
                        {
                            "sample_id": sample_id,
                            "class_name": class_name,
                            "imagenet_name": display_name,
                            "predictor_layer": layer,
                            "predictor_mean_offdiagonal_cosine": predictor_self_mean,
                            "target_mean_offdiagonal_cosine": target_self_mean,
                            "predictor_minus_target": predictor_self_mean - target_self_mean,
                        }
                    )
                    geometry_key = (layer, class_name)
                    geometry_total = geometry_totals.setdefault(
                        geometry_key, {"images": 0.0, "predictor": 0.0, "target": 0.0}
                    )
                    geometry_total["images"] += 1
                    geometry_total["predictor"] += predictor_self_mean
                    geometry_total["target"] += target_self_mean

                    alignment_key = (layer, class_name)
                    alignment = alignment_totals.setdefault(
                        alignment_key,
                        {
                            "patches": 0.0, "top1": 0.0, "top2": 0.0, "top4": 0.0,
                            "cosine": 0.0, "margin": 0.0,
                        },
                    )
                    for patch_offset in range(n_patches):
                        rank = int(correct_ranks[patch_offset])
                        best_target_offset = int(best_targets[patch_offset])
                        cosine = float(diagonal[patch_offset])
                        margin = float(diagonal[patch_offset] - best_incorrect[patch_offset])
                        writer.writerow(
                            {
                                "sample_id": sample_id,
                                "class_name": class_name,
                                "imagenet_name": display_name,
                                "predictor_layer": layer,
                                "predicted_patch_offset": patch_offset,
                                "predicted_patch_id": image_patch_ids[patch_offset],
                                "best_target_offset": best_target_offset,
                                "best_target_patch_id": image_patch_ids[best_target_offset],
                                "corresponding_target_rank": rank,
                                "top1_correct": int(rank <= 1),
                                "top2_correct": int(rank <= 2),
                                "top4_correct": int(rank <= 4),
                                "corresponding_cosine": cosine,
                                "best_incorrect_cosine": float(best_incorrect[patch_offset]),
                                "corresponding_margin_over_best_incorrect": margin,
                            }
                        )
                        alignment["patches"] += 1
                        alignment["top1"] += int(rank <= 1)
                        alignment["top2"] += int(rank <= 2)
                        alignment["top4"] += int(rank <= 4)
                        alignment["cosine"] += cosine
                        alignment["margin"] += margin

    alignment_class_rows: list[dict[str, Any]] = []
    for (layer, class_name), total in sorted(alignment_totals.items()):
        count = total["patches"]
        alignment_class_rows.append(
            {
                "predictor_layer": layer,
                "class_name": class_name,
                "imagenet_name": imagenet_display_name(class_name),
                "n_patches": int(count),
                "top1_rate": total["top1"] / count,
                "top2_rate": total["top2"] / count,
                "top4_rate": total["top4"] / count,
                "mean_corresponding_cosine": total["cosine"] / count,
                "mean_margin_over_best_incorrect": total["margin"] / count,
            }
        )
    geometry_class_rows: list[dict[str, Any]] = []
    for (layer, class_name), total in sorted(geometry_totals.items()):
        count = total["images"]
        predictor_mean = total["predictor"] / count
        target_mean = total["target"] / count
        geometry_class_rows.append(
            {
                "predictor_layer": layer,
                "class_name": class_name,
                "imagenet_name": imagenet_display_name(class_name),
                "n_images": int(count),
                "predictor_mean_offdiagonal_cosine": predictor_mean,
                "target_mean_offdiagonal_cosine": target_mean,
                "predictor_minus_target": predictor_mean - target_mean,
            }
        )
    write_csv(output_dir / "patch_geometry_by_image.csv", geometry_rows)
    write_csv(output_dir / "patch_alignment_class_metrics.csv", alignment_class_rows)
    write_csv(output_dir / "patch_geometry_class_metrics.csv", geometry_class_rows)


def export_pooled_distribution_stats(
    output_dir: Path,
    data: PooledData,
    predictor_layers: Sequence[int],
    vit_layers: Sequence[int],
) -> None:
    """Population spread of the pooled vectors, layer by layer.

    For each representation this records how the mean-pooled image vectors are
    distributed across the 1000 images: their mean/std magnitude and the total
    and mean per-dimension variance. This gives a spread trajectory from the
    first predictor layer to the last, with the target-encoder final layer as
    the reference row. Raw per-dimension mean/std vectors are dumped to a .pt so
    any other distribution question can be answered without rerunning capture.
    """
    rows: list[dict[str, Any]] = []
    raw: dict[str, torch.Tensor] = {}

    def add(name: str, vectors: np.ndarray) -> None:
        vectors = np.asarray(vectors, dtype=np.float64)
        per_dim_std = vectors.std(axis=0)
        per_dim_mean = vectors.mean(axis=0)
        norms = np.linalg.norm(vectors, axis=1)
        rows.append(
            {
                "representation": name,
                "n_images": int(vectors.shape[0]),
                "n_dims": int(vectors.shape[1]),
                "mean_norm": float(norms.mean()),
                "std_norm": float(norms.std()),
                "mean_per_dim_std": float(per_dim_std.mean()),
                "total_variance": float((per_dim_std ** 2).sum()),
            }
        )
        raw[f"{name}__per_dim_mean"] = torch.from_numpy(per_dim_mean)
        raw[f"{name}__per_dim_std"] = torch.from_numpy(per_dim_std)

    for index, layer in enumerate(predictor_layers):
        add(f"predictor_{layer:02d}", data.predictor[:, index])
    add("target_final", data.target)
    if data.vit is not None:
        for index, layer in enumerate(vit_layers):
            add(f"vit_{layer:02d}", data.vit[:, index])

    write_csv(output_dir / "pooled_distribution_stats.csv", rows)
    torch.save(raw, output_dir / "pooled_distribution_raw.pt")


def export_readable_metrics(
    capture_dir: Path,
    output_dir: Path,
    predictor_layers: Sequence[int],
    target_patch_ids: Sequence[int] | None,
) -> None:
    corresponding_path = output_dir / "corresponding_patch_metrics.csv"
    pooled_path = output_dir / "pooled_cosine_by_image.csv"
    with corresponding_path.open("w", newline="", encoding="utf-8") as corresponding_handle, pooled_path.open(
        "w", newline="", encoding="utf-8"
    ) as pooled_handle:
        corresponding_writer = csv.DictWriter(
            corresponding_handle,
            fieldnames=("sample_id", "predictor_layer", "target_offset", "patch_id", "cosine", "l1", "l2"),
        )
        pooled_writer = csv.DictWriter(
            pooled_handle, fieldnames=("sample_id", "predictor_layer", "pooled_cosine")
        )
        corresponding_writer.writeheader()
        pooled_writer.writeheader()
        for path in sorted((capture_dir / "metric_shards").glob("shard_*.pt")):
            shard = torch.load(path, map_location="cpu", weights_only=True)
            for batch_index, sample_id_tensor in enumerate(shard["sample_ids"]):
                sample_id = int(sample_id_tensor)
                image_patch_ids = (
                    [int(value) for value in shard["target_patch_ids"][batch_index]]
                    if "target_patch_ids" in shard
                    else list(target_patch_ids or ())
                )
                for layer_offset, layer in enumerate(predictor_layers):
                    pooled_writer.writerow(
                        {
                            "sample_id": sample_id,
                            "predictor_layer": layer,
                            "pooled_cosine": float(shard["pooled_cosine"][batch_index, layer_offset]),
                        }
                    )
                    for target_offset, patch_id in enumerate(image_patch_ids):
                        corresponding_writer.writerow(
                            {
                                "sample_id": sample_id,
                                "predictor_layer": layer,
                                "target_offset": target_offset,
                                "patch_id": patch_id,
                                "cosine": float(shard["corresponding_cosine"][batch_index, layer_offset, target_offset]),
                                "l1": float(shard["corresponding_l1"][batch_index, layer_offset, target_offset]),
                                "l2": float(shard["corresponding_l2"][batch_index, layer_offset, target_offset]),
                            }
                        )


def combine_condition_outputs(capture_dirs: Sequence[Path]) -> None:
    """Create small suite-level tables while leaving all raw values per condition."""
    if len(capture_dirs) < 2:
        return
    suite_dir = capture_dirs[0].parent
    if any(path.parent != suite_dir for path in capture_dirs):
        return
    output_dir = suite_dir / "analysis"
    output_dir.mkdir(exist_ok=True)

    for filename in ("probe_layer_metrics.csv", "probe_class_metrics.csv"):
        combined: list[dict[str, Any]] = []
        for capture_dir in capture_dirs:
            path = capture_dir / "analysis" / filename
            if not path.is_file():
                continue
            with path.open(newline="", encoding="utf-8") as handle:
                for row in csv.DictReader(handle):
                    combined.append({"condition": capture_dir.name, **row})
        write_csv(output_dir / f"combined_{filename}", combined)

    pooled_rows: list[dict[str, Any]] = []
    for capture_dir in capture_dirs:
        path = capture_dir / "analysis" / "pooled_cosine_by_image.csv"
        if not path.is_file():
            continue
        with path.open(newline="", encoding="utf-8") as handle:
            pooled_rows.extend({"condition": capture_dir.name, **row} for row in csv.DictReader(handle))
    write_csv(output_dir / "combined_pooled_cosine_by_image.csv", pooled_rows)

    reference_name = str(ANALYSIS_CONFIG["comparison_reference"])
    reference = {
        (int(row["sample_id"]), int(row["predictor_layer"])): float(row["pooled_cosine"])
        for row in pooled_rows
        if row["condition"] == reference_name
    }
    delta_rows: list[dict[str, Any]] = []
    if reference:
        for row in pooled_rows:
            key = (int(row["sample_id"]), int(row["predictor_layer"]))
            if row["condition"] == reference_name or key not in reference:
                continue
            value = float(row["pooled_cosine"])
            delta_rows.append(
                {
                    "condition": row["condition"],
                    "reference_condition": reference_name,
                    "sample_id": key[0],
                    "predictor_layer": key[1],
                    "pooled_cosine": value,
                    "reference_pooled_cosine": reference[key],
                    "delta": value - reference[key],
                }
            )
    write_csv(output_dir / "paired_pooled_cosine_deltas.csv", delta_rows)


def main() -> None:
    _, project_config = load_project_config(
        "Analyze an I-JEPA capture suite using its canonical YAML config."
    )
    ANALYSIS_CONFIG.clear()
    ANALYSIS_CONFIG.update(project_config["analysis"])
    capture_dirs = resolve_capture_dirs()
    for capture_dir in capture_dirs:
        metadata = json.loads((capture_dir / "capture_metadata.json").read_text(encoding="utf-8"))
        manifest = read_manifest(capture_dir / "dataset_manifest.csv")
        data = load_pooled_representations(capture_dir)
        expected_ids = np.arange(len(manifest))
        if not np.array_equal(data.sample_ids, expected_ids):
            raise RuntimeError("Representation shard sample IDs do not match the manifest.")

        output_dir = capture_dir / "analysis"
        output_dir.mkdir(exist_ok=True)
        layers = [int(layer) for layer in metadata["predictor_layers"]]
        vit_layers = [int(layer) for layer in metadata.get("vit_layers", [])]
        static_ids = metadata.get("target_patch_ids")
        target_patch_ids = [int(patch) for patch in static_ids] if static_ids is not None else None
        export_pooled_distribution_stats(output_dir, data, layers, vit_layers)
        if ANALYSIS_CONFIG["run_probe"]:
            if ANALYSIS_CONFIG["probe_mode"] == "cross_validation":
                run_cross_validated_probes(output_dir, manifest, data, layers, vit_layers)
            elif ANALYSIS_CONFIG["probe_mode"] == "fixed_split":
                run_probes(output_dir, manifest, data, layers, vit_layers)
            else:
                raise ValueError("probe_mode must be cross_validation or fixed_split.")
        if ANALYSIS_CONFIG["run_patch_alignment"]:
            run_patch_alignment(capture_dir, output_dir, manifest, layers, target_patch_ids)
        if ANALYSIS_CONFIG["export_readable_raw_metrics"]:
            export_readable_metrics(capture_dir, output_dir, layers, target_patch_ids)
        print(f"Analysis complete: {output_dir.resolve()}")
    combine_condition_outputs(capture_dirs)


if __name__ == "__main__":
    main()
