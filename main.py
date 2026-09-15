"""Capture raw layerwise I-JEPA representations and patch metrics.

This file only performs model inference and records outputs. It does not train
probes, aggregate across images, or make plots. Those operations belong in
analyze.py so analyses can change without rerunning the frozen model.

For every image this saves visible context tokens, final target tokens, raw and
projected predictor tokens at each selected predictor layer, raw corresponding-
patch metrics, raw all-pairs patch metrics, and mean-pooled cosine values.

The target encoder is held at its final layer. Predictor depth, mask condition,
input perturbation, and an optional ViT reference are explicit experimental axes.
Run ``python main.py --config configs/<experiment>.yaml``.
"""

from __future__ import annotations

import csv
import json
import math
import random
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from world_model_lens.hub.model_hub import ModelHub

from config_utils import load_project_config

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}
IMAGENET_MEAN = np.asarray((0.485, 0.456, 0.406), dtype=np.float32)
IMAGENET_STD = np.asarray((0.229, 0.224, 0.225), dtype=np.float32)

def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def validate_args(args: SimpleNamespace) -> None:
    if args.manifest is None and args.imagenet_root is None:
        raise ValueError("Set imagenet_root or manifest.")
    if args.manifest is not None and not args.manifest.is_file():
        raise FileNotFoundError(args.manifest)
    if args.manifest is None and not args.imagenet_root.is_dir():
        raise FileNotFoundError(args.imagenet_root)
    if args.checkpoint is not None and not args.checkpoint.is_file():
        raise FileNotFoundError(args.checkpoint)
    if args.num_classes < 2 or args.samples_per_class < 2:
        raise ValueError("Need at least two classes and two samples per class.")
    if not 0.0 < args.test_fraction < 1.0:
        raise ValueError("test_fraction must be between zero and one.")
    if args.batch_size < 1:
        raise ValueError("batch_size must be positive.")
    enabled = [condition for condition in args.conditions if condition.get("enabled", False)]
    if not enabled:
        raise ValueError("Enable at least one capture condition.")
    names = [str(condition.get("name", "")) for condition in enabled]
    if any(not name for name in names) or len(names) != len(set(names)):
        raise ValueError("Every enabled condition needs a unique non-empty name.")
    if any(Path(name).name != name or any(mark in name for mark in ("/", "\\")) for name in names):
        raise ValueError("Condition names must be plain directory names.")
    for condition in enabled:
        height, width = condition.get("mask", (0, 0))
        if not (1 <= int(height) <= 16 and 1 <= int(width) <= 16):
            raise ValueError(f"Invalid mask in condition {condition['name']}: {condition.get('mask')}")
        if int(height) * int(width) >= 256:
            raise ValueError(f"Condition {condition['name']} must leave at least one context patch.")
        if condition.get("position", "center") not in {"center", "random", "object"}:
            raise ValueError(f"Invalid mask position in condition {condition['name']}.")
        if condition.get("augmentation", "clean") not in {"clean", "dim", "noise", "half_swap"}:
            raise ValueError(f"Invalid augmentation in condition {condition['name']}.")


def manifest_label(row: dict[str, Any]) -> str:
    for key in ("class_name", "label", "class_id", "synset"):
        if key in row:
            return str(row[key])
    return Path(str(row["path"])).parent.name


def load_samples(args: SimpleNamespace) -> list[dict[str, Any]]:
    if args.manifest is not None:
        payload = json.loads(args.manifest.read_text(encoding="utf-8"))
        if not isinstance(payload, list) or not payload:
            raise ValueError("Manifest must be a non-empty JSON list.")
        samples: list[dict[str, Any]] = []
        for row in payload:
            if not isinstance(row, dict) or "path" not in row:
                raise ValueError("Every manifest row must contain path.")
            path = Path(str(row["path"]))
            if not path.is_absolute():
                path = (args.manifest.parent / path).resolve()
            if not path.is_file():
                raise FileNotFoundError(path)
            sample = {"path": str(path), "class_name": manifest_label(row)}
            if "bbox" in row:
                bbox = row["bbox"]
                if not isinstance(bbox, list) or len(bbox) != 4:
                    raise ValueError("bbox must be [xmin, ymin, xmax, ymax].")
                sample["bbox"] = [float(value) for value in bbox]
            elif all(key in row for key in ("xmin", "ymin", "xmax", "ymax")):
                sample["bbox"] = [float(row[key]) for key in ("xmin", "ymin", "xmax", "ymax")]
            samples.append(sample)
        return samples

    rng = random.Random(args.seed)
    eligible: list[tuple[str, list[Path]]] = []
    for class_dir in sorted(path for path in args.imagenet_root.iterdir() if path.is_dir()):
        images = sorted(
            path
            for path in class_dir.rglob("*")
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        )
        if len(images) >= args.samples_per_class:
            eligible.append((class_dir.name, images))
    if len(eligible) < args.num_classes:
        raise ValueError(
            f"Only {len(eligible)} classes have at least {args.samples_per_class} images; "
            f"requested {args.num_classes}."
        )
    rng.shuffle(eligible)
    samples = []
    for class_name, images in eligible[: args.num_classes]:
        images = list(images)
        rng.shuffle(images)
        samples.extend(
            {"path": str(path.resolve()), "class_name": class_name}
            for path in images[: args.samples_per_class]
        )
    rng.shuffle(samples)
    return samples


def encode_labels(samples: Sequence[dict[str, Any]]) -> tuple[np.ndarray, list[str]]:
    class_names = sorted({sample["class_name"] for sample in samples})
    mapping = {name: index for index, name in enumerate(class_names)}
    labels = np.asarray([mapping[sample["class_name"]] for sample in samples], dtype=np.int64)
    return labels, class_names


def stratified_split(labels: np.ndarray, test_fraction: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    train: list[int] = []
    test: list[int] = []
    for label in np.unique(labels):
        indices = np.flatnonzero(labels == label)
        rng.shuffle(indices)
        n_test = max(1, int(round(len(indices) * test_fraction)))
        n_test = min(n_test, len(indices) - 1)
        test.extend(indices[:n_test].tolist())
        train.extend(indices[n_test:].tolist())
    rng.shuffle(train)
    rng.shuffle(test)
    return np.asarray(train, dtype=np.int64), np.asarray(test, dtype=np.int64)


def standardize_pil(path: str | Path, image_size: int = 224) -> Image.Image:
    image = Image.open(path).convert("RGB")
    width, height = image.size
    scale = image_size / min(width, height)
    image = image.resize((round(width * scale), round(height * scale)), Image.Resampling.BICUBIC)
    left = (image.width - image_size) // 2
    top = (image.height - image_size) // 2
    return image.crop((left, top, left + image_size, top + image_size))


def pil_to_normalized_tensor(image: Image.Image) -> torch.Tensor:
    array = np.asarray(image, dtype=np.float32) / 255.0
    array = (array - IMAGENET_MEAN) / IMAGENET_STD
    return torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0)


def augment_image(
    image: Image.Image, condition: dict[str, Any], sample_id: int, seed: int
) -> Image.Image:
    """Apply a deterministic perturbation in unnormalized pixel space."""
    kind = str(condition.get("augmentation", "clean"))
    if kind == "clean":
        return image.copy()
    array = np.asarray(image, dtype=np.float32) / 255.0
    if kind == "dim":
        array = array * float(condition.get("dim_factor", 0.5))
    elif kind == "noise":
        rng = np.random.default_rng(seed + sample_id * 1_000_003)
        array = np.clip(
            array + rng.normal(0.0, float(condition.get("noise_std", 0.1)), array.shape),
            0.0,
            1.0,
        )
    elif kind == "half_swap":
        split = array.shape[1] // 2
        array = np.concatenate((array[:, split:, :], array[:, :split, :]), axis=1)
    else:
        raise ValueError(f"Unknown augmentation: {kind}")
    return Image.fromarray(np.rint(array * 255.0).astype(np.uint8), mode="RGB")


def load_condition_images(
    sample: dict[str, Any], condition: dict[str, Any], sample_id: int, seed: int
) -> tuple[torch.Tensor, torch.Tensor]:
    clean = standardize_pil(sample["path"], 224)
    perturbed = augment_image(clean, condition, sample_id, seed)
    scope = str(condition.get("augmentation_scope", "context_only"))
    if scope not in {"context_only", "both"}:
        raise ValueError("augmentation_scope must be context_only or both.")
    context_image = perturbed
    target_image = perturbed if scope == "both" else clean
    return pil_to_normalized_tensor(context_image), pil_to_normalized_tensor(target_image)


def resolve_device_precision(args: SimpleNamespace) -> tuple[torch.device, torch.dtype]:
    device_name = "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
    if device_name == "auto":
        device_name = "cpu"
    device = torch.device(device_name)
    precision = args.precision
    if precision == "auto":
        precision = "fp16" if device.type == "cuda" else "fp32"
    if precision == "fp16" and device.type != "cuda":
        raise ValueError("fp16 is only supported on CUDA.")
    return device, torch.float16 if precision == "fp16" else torch.float32


def load_world_model(args: SimpleNamespace, device: torch.device, dtype: torch.dtype) -> Any:
    if args.checkpoint is not None:
        adapter = ModelHub._load_ijepa(str(args.checkpoint), device=str(device))
    else:
        adapter = ModelHub.load(args.model_name, device=str(device))
    adapter = adapter.half() if dtype == torch.float16 else adapter.float()
    adapter.eval()
    for parameter in adapter.parameters():
        parameter.requires_grad_(False)
    return adapter


def transformed_bbox_center(sample: dict[str, Any], image_size: int = 224) -> tuple[float, float]:
    """Map an original-image bbox center through resize-and-center-crop."""
    if "bbox" not in sample:
        raise ValueError(
            f"Object-aligned condition needs bbox for {sample['path']}; use a JSON manifest "
            "with bbox: [xmin, ymin, xmax, ymax]."
        )
    with Image.open(sample["path"]) as image:
        original_width, original_height = image.size
    scale = image_size / min(original_width, original_height)
    resized_width = round(original_width * scale)
    resized_height = round(original_height * scale)
    crop_left = (resized_width - image_size) // 2
    crop_top = (resized_height - image_size) // 2
    xmin, ymin, xmax, ymax = (float(value) for value in sample["bbox"])
    transformed_xmin = xmin * scale - crop_left
    transformed_xmax = xmax * scale - crop_left
    transformed_ymin = ymin * scale - crop_top
    transformed_ymax = ymax * scale - crop_top
    visible_xmin = float(np.clip(transformed_xmin, 0, image_size))
    visible_xmax = float(np.clip(transformed_xmax, 0, image_size))
    visible_ymin = float(np.clip(transformed_ymin, 0, image_size))
    visible_ymax = float(np.clip(transformed_ymax, 0, image_size))
    if visible_xmax <= visible_xmin or visible_ymax <= visible_ymin:
        raise ValueError(f"bbox is outside the 224-pixel center crop for {sample['path']}.")
    center_x = (visible_xmin + visible_xmax) * 0.5
    center_y = (visible_ymin + visible_ymax) * 0.5
    return center_x, center_y


def build_mask(
    adapter: Any,
    condition: dict[str, Any],
    sample: dict[str, Any],
    sample_id: int,
    seed: int,
) -> tuple[list[int], list[int], int]:
    num_patches = int(adapter.context_encoder.patch_embed.n_patches)
    grid = int(math.sqrt(num_patches))
    height, width = (int(value) for value in condition["mask"])
    if grid * grid != num_patches or height > grid or width > grid:
        raise ValueError(f"Invalid {height} by {width} mask for {num_patches} patches.")
    position = str(condition.get("position", "center"))
    if position == "center":
        start_row = (grid - height) // 2
        start_col = (grid - width) // 2
    elif position == "random":
        # The same image gets the same random mask across repeated runs. Object
        # and random conditions with the same shape therefore differ only in position.
        rng = random.Random(seed + sample_id * 1_000_003 + height * 1009 + width)
        start_row = rng.randrange(grid - height + 1)
        start_col = rng.randrange(grid - width + 1)
    elif position == "object":
        center_x, center_y = transformed_bbox_center(sample)
        center_col = int(np.clip(math.floor(center_x / 224 * grid), 0, grid - 1))
        center_row = int(np.clip(math.floor(center_y / 224 * grid), 0, grid - 1))
        start_row = int(np.clip(center_row - height // 2, 0, grid - height))
        start_col = int(np.clip(center_col - width // 2, 0, grid - width))
    else:
        raise ValueError(f"Unknown mask position: {position}")
    target = sorted(
        row * grid + col
        for row in range(start_row, start_row + height)
        for col in range(start_col, start_col + width)
    )
    target_set = set(target)
    context = [patch for patch in range(num_patches) if patch not in target_set]
    return context, target, grid


def load_vit_baseline(
    config: dict[str, Any], device: torch.device, dtype: torch.dtype
) -> tuple[Any, list[int]] | tuple[None, list[int]]:
    if not config.get("enabled", False):
        return None, []
    checkpoint = config.get("checkpoint")
    model = torch.hub.load(
        str(config.get("repo", "facebookresearch/dinov2")),
        str(config.get("model", "dinov2_vitb14")),
        pretrained=checkpoint is None,
        source=str(config.get("source", "github")),
    )
    if checkpoint is not None:
        checkpoint_path = Path(checkpoint)
        if not checkpoint_path.is_file():
            raise FileNotFoundError(checkpoint_path)
        payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        if isinstance(payload, dict):
            for key in ("model", "state_dict", "teacher"):
                if key in payload and isinstance(payload[key], dict):
                    payload = payload[key]
                    break
        cleaned = {str(key).removeprefix("module."): value for key, value in payload.items()}
        for prefix in ("teacher.backbone.", "backbone."):
            backbone = {
                key.removeprefix(prefix): value
                for key, value in cleaned.items()
                if key.startswith(prefix)
            }
            if backbone:
                cleaned = backbone
                break
        model.load_state_dict(cleaned, strict=True)
    model = model.to(device=device)
    model = model.half() if dtype == torch.float16 else model.float()
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    blocks = getattr(model, "blocks", None)
    if blocks is None:
        raise AttributeError("ViT baseline must expose transformer blocks as model.blocks.")
    return model, resolve_layers(str(config.get("layers", "all")), len(blocks))


@torch.no_grad()
def capture_vit_tokens(
    model: Any,
    layers: Sequence[int],
    images: torch.Tensor,
    target_ids: Sequence[int],
    grid: int,
) -> torch.Tensor:
    captured: dict[int, torch.Tensor] = {}

    def hook_for(layer: int):
        def hook(_module: Any, _inputs: Any, output: Any) -> None:
            tensor = output[0] if isinstance(output, tuple) else output
            captured[layer] = tensor.detach()

        return hook

    handles = [model.blocks[layer].register_forward_hook(hook_for(layer)) for layer in layers]
    try:
        model(images)
    finally:
        for handle in handles:
            handle.remove()
    needed = grid * grid
    selected: list[torch.Tensor] = []
    for layer in layers:
        tokens = captured[layer]
        if tokens.ndim != 3 or tokens.shape[1] < needed:
            raise RuntimeError(f"Unexpected ViT block output shape at layer {layer}: {tokens.shape}")
        # DINOv2 prefixes class/register tokens; spatial patch tokens are last.
        patch_tokens = tokens[:, -needed:, :]
        if hasattr(model, "norm"):
            patch_tokens = model.norm(patch_tokens)
        selected.append(patch_tokens[:, list(target_ids), :])
    return torch.stack(selected, dim=1)


def resolve_layers(spec: str, depth: int) -> list[int]:
    if spec.strip().lower() == "all":
        return list(range(depth))
    layers = sorted({int(value.strip()) for value in spec.split(",") if value.strip()})
    invalid = [layer for layer in layers if layer < 0 or layer >= depth]
    if not layers or invalid:
        raise ValueError(f"Invalid predictor layers {invalid or layers}; depth is {depth}.")
    return layers


def predictor_head(predictor: Any) -> tuple[Any, Any]:
    norm = getattr(predictor, "norm", None) or getattr(predictor, "predictor_norm", None)
    projection = getattr(predictor, "predictor_project_back", None) or getattr(
        predictor, "predictor_proj", None
    )
    if norm is None or projection is None:
        raise AttributeError("Could not find the predictor's final norm/projection.")
    return norm, projection


def all_pairs_metrics(
    predicted: torch.Tensor, target: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return tensors indexed by image, layer, predicted patch, target patch."""
    cosine_layers: list[torch.Tensor] = []
    l1_layers: list[torch.Tensor] = []
    l2_layers: list[torch.Tensor] = []
    target_float = target.float()
    target_normalized = F.normalize(target_float, dim=-1)
    for layer_predictions in predicted.unbind(dim=1):
        prediction_float = layer_predictions.float()
        cosine_layers.append(F.normalize(prediction_float, dim=-1) @ target_normalized.transpose(1, 2))
        difference = prediction_float[:, :, None, :] - target_float[:, None, :, :]
        l1_layers.append(difference.abs().mean(dim=-1))
        l2_layers.append(torch.linalg.vector_norm(difference, dim=-1))
    return (
        torch.stack(cosine_layers, dim=1),
        torch.stack(l1_layers, dim=1),
        torch.stack(l2_layers, dim=1),
    )


def write_manifest(
    path: Path,
    samples: Sequence[dict[str, Any]],
    labels: np.ndarray,
    train_indices: np.ndarray,
    test_indices: np.ndarray,
) -> None:
    train_set = set(train_indices.tolist())
    test_set = set(test_indices.tolist())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=("sample_id", "path", "class_name", "label_id", "split", "bbox"),
        )
        writer.writeheader()
        for sample_id, sample in enumerate(samples):
            split = "train" if sample_id in train_set else "test" if sample_id in test_set else ""
            writer.writerow(
                {
                    "sample_id": sample_id,
                    "path": sample["path"],
                    "class_name": sample["class_name"],
                    "label_id": int(labels[sample_id]),
                    "split": split,
                    "bbox": json.dumps(sample.get("bbox")) if "bbox" in sample else "",
                }
            )


@torch.no_grad()
def capture_condition(
    args: SimpleNamespace,
    adapter: Any,
    vit_model: Any,
    vit_layers: Sequence[int],
    samples: Sequence[dict[str, Any]],
    condition: dict[str, Any],
    run_dir: Path,
) -> None:
    device, dtype = resolve_device_precision(args)
    dynamic_masks = condition.get("position", "center") in {"random", "object"}
    first_context_ids, first_target_ids, grid = build_mask(
        adapter, condition, samples[0], 0, args.seed
    )
    depth = len(adapter.predictor.blocks)
    layers = resolve_layers(args.layers, depth)
    norm, projection = predictor_head(adapter.predictor)
    effective_batch_size = 1 if dynamic_masks else args.batch_size

    representation_dir = run_dir / "representation_shards"
    metric_dir = run_dir / "metric_shards"
    representation_dir.mkdir()
    metric_dir.mkdir()

    metadata = {
        "format_version": 2,
        "condition": condition,
        "target_encoder_layer": "final",
        "predictor_layers": layers,
        "predictor_capture_point": "post_block",
        "vit_layers": list(vit_layers),
        "grid_size": grid,
        "dynamic_patch_ids": dynamic_masks,
        "context_patch_ids": None if dynamic_masks else first_context_ids,
        "target_patch_ids": None if dynamic_masks else first_target_ids,
        "tensor_axes": {
            "context_tokens": ["image", "context_patch", "encoder_feature"],
            "target_tokens": ["image", "target_patch", "encoder_feature"],
            "predictor_hidden": ["image", "predictor_layer", "target_patch", "predictor_feature"],
            "predictor_projected": ["image", "predictor_layer", "target_patch", "encoder_feature"],
            "corresponding_metrics": ["image", "predictor_layer", "target_patch"],
            "all_pairs_metrics": ["image", "predictor_layer", "predicted_patch", "target_patch"],
            "predictor_self_cosine": ["image", "predictor_layer", "patch", "patch"],
            "target_self_cosine": ["image", "patch", "patch"],
            "pooled_cosine": ["image", "predictor_layer"],
            "target_patch_ids_per_image": ["image", "target_patch"],
            "vit_tokens": ["image", "vit_layer", "target_patch", "vit_feature"],
        },
        "distance_definitions": {
            "l1": "mean absolute coordinate difference",
            "l2": "Euclidean distance",
        },
        "device": str(device),
        "model_dtype": str(dtype),
    }
    (run_dir / "capture_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    max_final_projection_error = 0.0
    for shard_index, start in enumerate(range(0, len(samples), effective_batch_size)):
        batch_samples = samples[start : start + effective_batch_size]
        sample_ids = torch.arange(start, start + len(batch_samples), dtype=torch.int64)
        masks = [
            build_mask(adapter, condition, sample, start + offset, args.seed)
            for offset, sample in enumerate(batch_samples)
        ]
        target_id_rows = [mask[1] for mask in masks]
        if any(row != target_id_rows[0] for row in target_id_rows[1:]):
            raise RuntimeError("A batch contains different patch masks; this condition must use batch size one.")
        context_ids, target_ids, _ = masks[0]
        image_pairs = [
            load_condition_images(sample, condition, start + offset, args.seed)
            for offset, sample in enumerate(batch_samples)
        ]
        context_images = torch.cat([pair[0] for pair in image_pairs]).to(device=device, dtype=dtype)
        target_images = torch.cat([pair[1] for pair in image_pairs]).to(device=device, dtype=dtype)
        captured: dict[int, torch.Tensor] = {}

        def hook_for(layer: int):
            def hook(_module: Any, _inputs: Any, output: torch.Tensor) -> None:
                captured[layer] = output.detach()

            return hook

        handles = [
            adapter.predictor.blocks[layer].hook_resid_post.register_forward_hook(hook_for(layer))
            for layer in layers
        ]
        try:
            context_tokens = adapter.context_encoder(context_images, patch_ids=context_ids)
            final_prediction = adapter.predictor(context_tokens, context_ids, target_ids)
            target_tokens = adapter.target_encoder(target_images)[:, target_ids, :]
        finally:
            for handle in handles:
                handle.remove()

        hidden = torch.stack(
            [captured[layer][:, len(context_ids) :, :] for layer in layers], dim=1
        )
        projected = torch.stack(
            [projection(norm(hidden[:, index])) for index in range(len(layers))], dim=1
        )
        if layers[-1] == depth - 1:
            max_final_projection_error = max(
                max_final_projection_error,
                float((projected[:, -1].float() - final_prediction.float()).abs().max()),
            )

        prediction_float = projected.float()
        target_float = target_tokens.float()
        corresponding_cosine = F.cosine_similarity(
            prediction_float, target_float[:, None, :, :], dim=-1
        )
        corresponding_difference = prediction_float - target_float[:, None, :, :]
        corresponding_l1 = corresponding_difference.abs().mean(dim=-1)
        corresponding_l2 = torch.linalg.vector_norm(corresponding_difference, dim=-1)
        pair_cosine, pair_l1, pair_l2 = all_pairs_metrics(projected, target_tokens)
        normalized_predictions = F.normalize(prediction_float, dim=-1)
        predictor_self_cosine = normalized_predictions @ normalized_predictions.transpose(-1, -2)
        normalized_targets = F.normalize(target_float, dim=-1)
        target_self_cosine = normalized_targets @ normalized_targets.transpose(-1, -2)
        pooled_cosine = F.cosine_similarity(
            prediction_float.mean(dim=2), target_float.mean(dim=1)[:, None, :], dim=-1
        )

        representation_payload: dict[str, torch.Tensor] = {
            "sample_ids": sample_ids,
            "target_patch_ids": torch.as_tensor(target_id_rows, dtype=torch.int64),
            "target_tokens": target_tokens.detach().cpu(),
            "predictor_projected": projected.detach().cpu(),
        }
        if vit_model is not None:
            vit_input = target_images if args.vit_baseline.get("input", "target") == "target" else context_images
            representation_payload["vit_tokens"] = capture_vit_tokens(
                vit_model, vit_layers, vit_input, target_ids, grid
            ).detach().cpu()
        if args.save_context_tokens:
            representation_payload["context_tokens"] = context_tokens.detach().cpu()
        if args.save_predictor_hidden:
            representation_payload["predictor_hidden"] = hidden.detach().cpu()
        torch.save(representation_payload, representation_dir / f"shard_{shard_index:04d}.pt")
        torch.save(
            {
                "sample_ids": sample_ids,
                "target_patch_ids": torch.as_tensor(target_id_rows, dtype=torch.int64),
                "corresponding_cosine": corresponding_cosine.cpu(),
                "corresponding_l1": corresponding_l1.cpu(),
                "corresponding_l2": corresponding_l2.cpu(),
                "all_pairs_cosine": pair_cosine.cpu(),
                "all_pairs_l1": pair_l1.cpu(),
                "all_pairs_l2": pair_l2.cpu(),
                "predictor_self_cosine": predictor_self_cosine.cpu(),
                "target_self_cosine": target_self_cosine.cpu(),
                "pooled_cosine": pooled_cosine.cpu(),
            },
            metric_dir / f"shard_{shard_index:04d}.pt",
        )
        print(f"Captured {start + len(batch_samples)}/{len(samples)}", flush=True)

    summary = {
        "n_images": len(samples),
        "n_predictor_layers": len(layers),
        "n_context_patches": len(context_ids),
        "n_target_patches": len(target_ids),
        "n_shards": math.ceil(len(samples) / effective_batch_size),
        "effective_batch_size": effective_batch_size,
        "condition_name": condition["name"],
        "target_encoder_layer": "final",
        "max_final_projection_error": max_final_projection_error,
    }
    (run_dir / "capture_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


def json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    return value


def main() -> None:
    config_path, project_config = load_project_config(
        "Capture I-JEPA representations for one YAML experiment suite."
    )
    args = SimpleNamespace(**project_config["capture"])
    validate_args(args)
    seed_everything(args.seed)
    samples = load_samples(args)
    if any(
        condition.get("enabled", False) and condition.get("position") == "object"
        for condition in args.conditions
    ):
        missing_boxes = [sample["path"] for sample in samples if "bbox" not in sample]
        if missing_boxes:
            raise ValueError(
                "Object-aligned masks require bbox annotations for every sample; "
                f"first missing bbox: {missing_boxes[0]}"
            )
    labels, class_names = encode_labels(samples)
    train_indices, test_indices = stratified_split(labels, args.test_fraction, args.seed)
    print(
        f"Dataset: {len(samples)} images, {len(class_names)} classes, "
        f"{len(train_indices)} train, {len(test_indices)} test.",
        flush=True,
    )
    if args.check_only:
        print("Dataset check complete.")
        return

    device, dtype = resolve_device_precision(args)
    adapter = load_world_model(args, device, dtype)
    vit_model, vit_layers = load_vit_baseline(args.vit_baseline, device, dtype)
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    suite_dir = args.output_root / project_config["experiment_name"] / run_id
    suite_dir.mkdir(parents=True, exist_ok=False)
    resolved = json_safe(project_config)
    resolved.update(
        {
            "source_config": str(config_path),
            "run_id": run_id,
            "device_resolved": str(device),
            "dtype_resolved": str(dtype),
        }
    )
    (suite_dir / "config.json").write_text(json.dumps(resolved, indent=2), encoding="utf-8")
    enabled_conditions = [condition for condition in args.conditions if condition.get("enabled", False)]
    for condition in enabled_conditions:
        condition_dir = suite_dir / str(condition["name"])
        condition_dir.mkdir()
        write_manifest(
            condition_dir / "dataset_manifest.csv", samples, labels, train_indices, test_indices
        )
        capture_condition(
            args, adapter, vit_model, vit_layers, samples, condition, condition_dir
        )
        print(f"Condition complete: {condition_dir.resolve()}", flush=True)
    print(f"Raw capture suite complete: {suite_dir.resolve()}", flush=True)


if __name__ == "__main__":
    main()
