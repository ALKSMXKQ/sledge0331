from __future__ import annotations

import argparse
import csv
import json
import re
import shutil
from dataclasses import fields
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from omegaconf import OmegaConf
from tqdm import tqdm

from nuplan.planning.training.preprocessing.utils.feature_cache import FeatureCachePickle

from sledge.autoencoder.preprocessing.feature_builders.sledge.sledge_feature_processing import (
    sledge_raw_feature_processing,
)
from sledge.autoencoder.preprocessing.features.sledge_vector_feature import SledgeConfig, SledgeVector
from sledge.semantic_control.io import load_raw_scene, save_json


EDITED_METADATA_FILES = [
    "scenario_label.json",
    "edit_report.json",
    "edited_prompt_alignment.json",
    "summary.json",
]


def _to_builtin(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer, np.floating, np.bool_)):
        return value.item()
    if isinstance(value, tuple):
        return list(value)
    return value



def _search_best_config_dict(node: Any, target_keys: set[str]) -> Tuple[int, Optional[Dict[str, Any]]]:
    best_score = -1
    best_dict: Optional[Dict[str, Any]] = None

    if isinstance(node, dict):
        score = len(set(node.keys()) & target_keys)
        if score > best_score:
            best_score = score
            best_dict = node
        for value in node.values():
            child_score, child_dict = _search_best_config_dict(value, target_keys)
            if child_score > best_score:
                best_score = child_score
                best_dict = child_dict
    elif isinstance(node, list):
        for value in node:
            child_score, child_dict = _search_best_config_dict(value, target_keys)
            if child_score > best_score:
                best_score = child_score
                best_dict = child_dict
    return best_score, best_dict



def build_sledge_config(config_path: Optional[str]) -> SledgeConfig:
    default_cfg = {f.name: _to_builtin(getattr(SledgeConfig(), f.name)) for f in fields(SledgeConfig)}
    if not config_path:
        return SledgeConfig(**default_cfg)

    cfg = OmegaConf.to_container(OmegaConf.load(config_path), resolve=True)
    assert cfg is not None
    target_keys = set(default_cfg.keys())
    score, cfg_block = _search_best_config_dict(cfg, target_keys)
    if not cfg_block or score <= 0:
        print(f"[WARN] no SledgeConfig-like block found in {config_path}; using defaults")
        return SledgeConfig(**default_cfg)

    merged = dict(default_cfg)
    for key in default_cfg.keys():
        if key in cfg_block:
            merged[key] = _to_builtin(cfg_block[key])
    return SledgeConfig(**merged)



def safe_token_from_scene_path(scene_path: str) -> str:
    scene_path = str(scene_path)
    parts = Path(scene_path).parts[-4:-1]  # e.g. log/scene_type/token or log label/token
    base = "__".join(parts) if parts else Path(scene_path).stem
    base = re.sub(r"[^A-Za-z0-9_.-]+", "_", base).strip("_")
    return base[:180]



def read_manifest(manifest_path: Path) -> List[dict]:
    with open(manifest_path, "r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))



def store_vector(feature_store: FeatureCachePickle, out_dir: Path, vector: SledgeVector) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    feature_store.store_computed_feature_to_folder(out_dir / "sledge_vector", vector)
    return out_dir / "sledge_vector.gz"



def copy_if_exists(src: Path, dst: Path) -> None:
    if src.is_file():
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)



def convert_raw_to_vector(raw_path: Path, sledge_config: SledgeConfig) -> SledgeVector:
    raw_scene, _ = load_raw_scene(raw_path)
    vector, _ = sledge_raw_feature_processing(raw_scene, sledge_config)
    if not isinstance(vector, SledgeVector):
        raise TypeError(f"Expected SledgeVector, got {type(vector)}")
    return vector



def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True, help="Path to scenario_manifest.csv")
    parser.add_argument("--config", default=None, help="Hydra/yaml config used to extract SledgeConfig")
    parser.add_argument("--output-root", required=True, help="Root dir for paired vector caches")
    parser.add_argument("--accepted-only", action="store_true", help="Only convert accepted=True rows")
    parser.add_argument("--max-scenes", type=int, default=None)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--copy-edited-metadata", action="store_true")
    return parser



def main() -> None:
    args = build_argparser().parse_args()
    manifest_path = Path(args.manifest).resolve()
    output_root = Path(args.output_root).resolve()
    original_root = output_root / "original"
    edited_root = output_root / "edited"
    metadata_root = output_root / "metadata"
    for p in [original_root, edited_root, metadata_root]:
        p.mkdir(parents=True, exist_ok=True)

    rows = read_manifest(manifest_path)
    if args.accepted_only:
        rows = [r for r in rows if str(r.get("accepted", "")).lower() == "true"]
    if args.max_scenes is not None:
        rows = rows[: args.max_scenes]

    sledge_config = build_sledge_config(args.config)
    feature_store = FeatureCachePickle()

    paired_rows: List[Dict[str, Any]] = []
    failed_rows: List[Dict[str, Any]] = []

    for row in tqdm(rows, desc="build_paired_vector_caches", dynamic_ncols=True):
        try:
            original_scene_path = Path(row["scene_path"]).resolve()
            edited_raw_dir = Path(row["output_dir"]).resolve()
            edited_raw_path = edited_raw_dir / "sledge_raw.gz"
            scenario_type = str(row.get("scenario_type", "unknown"))
            severity_level = str(row.get("severity_level", ""))
            prompt = str(row.get("prompt", ""))
            accepted = str(row.get("accepted", ""))
            alignment = str(row.get("edited_alignment", ""))

            pair_id = safe_token_from_scene_path(original_scene_path)
            original_token_dir = original_root / "log" / "original" / pair_id
            edited_token_dir = edited_root / "log" / scenario_type / pair_id

            original_vector_path = original_token_dir / "sledge_vector.gz"
            edited_vector_path = edited_token_dir / "sledge_vector.gz"

            if args.skip_existing and original_vector_path.is_file() and edited_vector_path.is_file():
                pass
            else:
                original_vector = convert_raw_to_vector(original_scene_path, sledge_config)
                edited_vector = convert_raw_to_vector(edited_raw_path, sledge_config)
                store_vector(feature_store, original_token_dir, original_vector)
                store_vector(feature_store, edited_token_dir, edited_vector)

            pair_meta = {
                "pair_id": pair_id,
                "scenario_type": scenario_type,
                "severity_level": severity_level,
                "prompt": prompt,
                "accepted": accepted,
                "edited_alignment": alignment,
                "original_scene_path": str(original_scene_path),
                "edited_raw_dir": str(edited_raw_dir),
                "edited_raw_path": str(edited_raw_path),
                "original_vector_path": str(original_vector_path),
                "edited_vector_path": str(edited_vector_path),
                "sledge_config": {f.name: _to_builtin(getattr(sledge_config, f.name)) for f in fields(SledgeConfig)},
            }
            save_json(metadata_root / f"{pair_id}.json", pair_meta)

            if args.copy_edited_metadata:
                for name in EDITED_METADATA_FILES:
                    copy_if_exists(edited_raw_dir / name, edited_token_dir / name)

            paired_rows.append(
                {
                    "pair_id": pair_id,
                    "scenario_type": scenario_type,
                    "severity_level": severity_level,
                    "prompt": prompt,
                    "accepted": accepted,
                    "edited_alignment": alignment,
                    "original_scene_path": str(original_scene_path),
                    "edited_raw_path": str(edited_raw_path),
                    "original_vector_path": str(original_vector_path),
                    "edited_vector_path": str(edited_vector_path),
                }
            )
        except Exception as exc:
            failed_rows.append(
                {
                    "scene_path": row.get("scene_path", ""),
                    "output_dir": row.get("output_dir", ""),
                    "scenario_type": row.get("scenario_type", ""),
                    "error_type": type(exc).__name__,
                    "error": repr(exc),
                }
            )

    manifest_out = metadata_root / "paired_vector_manifest.csv"
    with open(manifest_out, "w", encoding="utf-8", newline="") as f:
        fieldnames = [
            "pair_id",
            "scenario_type",
            "severity_level",
            "prompt",
            "accepted",
            "edited_alignment",
            "original_scene_path",
            "edited_raw_path",
            "original_vector_path",
            "edited_vector_path",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(paired_rows)

    save_json(metadata_root / "paired_vector_failed_rows.json", failed_rows)
    save_json(
        metadata_root / "paired_vector_stats.json",
        {
            "total_manifest_rows": len(rows),
            "paired_success": len(paired_rows),
            "paired_failed": len(failed_rows),
            "original_cache_root": str(original_root),
            "edited_cache_root": str(edited_root),
            "manifest_path": str(manifest_path),
        },
    )

    print(f"[OK] paired_success={len(paired_rows)} failed={len(failed_rows)}")
    print(f"[OK] original cache: {original_root}")
    print(f"[OK] edited cache:   {edited_root}")
    print(f"[OK] manifest:       {manifest_out}")


if __name__ == "__main__":
    main()
