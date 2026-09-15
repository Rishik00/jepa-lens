"""Canonical YAML configuration loading shared by capture and analysis."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import yaml


def _resolve_optional_path(value: Any, project_root: Path) -> Path | None:
    if value is None:
        return None
    path = Path(str(value))
    return path if path.is_absolute() else (project_root / path).resolve()


def read_project_config(config_path: Path) -> dict[str, Any]:
    """Load and normalize one experiment YAML."""
    config_path = config_path.resolve()
    if not config_path.is_file():
        raise FileNotFoundError(config_path)

    with config_path.open(encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    if not isinstance(payload, dict):
        raise ValueError("The YAML root must be a mapping.")
    for section in ("experiment_name", "capture", "analysis"):
        if section not in payload:
            raise ValueError(f"Missing required YAML section: {section}")
    if not isinstance(payload["capture"], dict) or not isinstance(payload["analysis"], dict):
        raise ValueError("capture and analysis must be mappings.")

    project_root_value = payload.get("project_root", ".")
    project_root = (config_path.parent / str(project_root_value)).resolve()
    capture = dict(payload["capture"])
    for key in ("checkpoint", "imagenet_root", "manifest", "output_root"):
        capture[key] = _resolve_optional_path(capture.get(key), project_root)
    if capture["output_root"] is None:
        raise ValueError("capture.output_root is required.")
    vit = dict(capture.get("vit_baseline", {}))
    vit["checkpoint"] = _resolve_optional_path(vit.get("checkpoint"), project_root)
    if vit.get("source", "github") == "local" and vit.get("repo") is not None:
        vit["repo"] = str(_resolve_optional_path(vit["repo"], project_root))
    capture["vit_baseline"] = vit

    experiment_name = str(payload["experiment_name"])
    if not experiment_name or Path(experiment_name).name != experiment_name:
        raise ValueError("experiment_name must be a plain directory name.")

    analysis = dict(payload["analysis"])
    analysis["capture_dir"] = _resolve_optional_path(analysis.get("capture_dir"), project_root)
    analysis["capture_root"] = Path(capture["output_root"]) / experiment_name

    normalized = dict(payload)
    normalized["experiment_name"] = experiment_name
    normalized["capture"] = capture
    normalized["analysis"] = analysis
    normalized["_project_root"] = project_root
    return normalized


def load_project_config(description: str) -> tuple[Path, dict[str, Any]]:
    """Parse only --config and load the complete experiment definition."""
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--config", type=Path, required=True, help="Experiment YAML file")
    config_path = parser.parse_args().config.resolve()
    return config_path, read_project_config(config_path)
