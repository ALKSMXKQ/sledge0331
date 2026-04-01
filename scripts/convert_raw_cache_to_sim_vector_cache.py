from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import pickle
import shutil
from dataclasses import asdict, fields
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
from omegaconf import OmegaConf
from tqdm import tqdm

from nuplan.planning.training.preprocessing.utils.feature_cache import FeatureCachePickle

from sledge.autoencoder.preprocessing.feature_builders.sledge.sledge_feature_processing import (
    sledge_raw_feature_processing,
)
from sledge.autoencoder.preprocessing.features.sledge_vector_feature import SledgeConfig, SledgeVector
from sledge.semantic_control.io import load_raw_scene, save_json


SIBLING_METADATA_FILES = [
    "scenario_label.json",
    "edit_report.json",
    "edited_prompt_alignment.json",
    "summary.json",
]


def resolve_gz_path(path_like: str | Path) -> Path:
    path = Path(path_like)
    return path if path.suffix == ".gz" else path.with_suffix(".gz")


def iter_raw_paths(input_path: Path, glob_pattern: str) -> List[Path]:
    if input_path.is_file():
        if input_path.name.endswith("sledge_raw.gz"):
            return [input_path]
        raise ValueError(f"输入文件不是 sledge_raw.gz: {input_path}")
    return sorted(input_path.glob(glob_pattern))


def _to_builtin(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer, np.floating, np.bool_)):
        return value.item()
    if isinstance(value, tuple):
        return list(value)
    return value


def _search_best_config_dict(node: Any, target_keys: set[str]) -> Tuple[int, Optional[Dict[str, Any]]]:
    """递归搜索最像 SledgeConfig 的配置块。"""
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


class RawToVectorCacheConverter:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.input_path = Path(args.input).resolve()
        self.output_root = Path(args.output_root).resolve()
        self.output_root.mkdir(parents=True, exist_ok=True)
        self.raw_paths = iter_raw_paths(self.input_path, args.glob_pattern)
        if args.max_scenes is not None:
            self.raw_paths = self.raw_paths[: args.max_scenes]

        self.sledge_config = self._build_sledge_config(args.config)
        self.feature_store = FeatureCachePickle()
        self.rows: List[Dict[str, Any]] = []
        self.failed_rows: List[Dict[str, Any]] = []

    def _build_sledge_config(self, config_path: Optional[str]) -> SledgeConfig:
        default_cfg = {f.name: _to_builtin(getattr(SledgeConfig(), f.name)) for f in fields(SledgeConfig)}
        if not config_path:
            return SledgeConfig(**default_cfg)

        cfg = OmegaConf.to_container(OmegaConf.load(config_path), resolve=True)
        assert cfg is not None
        target_keys = set(default_cfg.keys())
        score, cfg_block = _search_best_config_dict(cfg, target_keys)
        if not cfg_block or score <= 0:
            print(f"[WARN] 未在配置文件中找到 SledgeConfig 对应字段，改用默认配置。config={config_path}")
            return SledgeConfig(**default_cfg)

        merged = dict(default_cfg)
        for key in default_cfg.keys():
            if key in cfg_block:
                merged[key] = _to_builtin(cfg_block[key])
        return SledgeConfig(**merged)

    def _relative_key(self, raw_path: Path) -> str:
        try:
            rel = raw_path.relative_to(self.input_path if self.input_path.is_dir() else self.input_path.parent)
        except Exception:
            rel = raw_path.name
        return str(rel)

    def _token_name(self, raw_path: Path) -> str:
        key = self._relative_key(raw_path)
        digest = hashlib.md5(key.encode("utf-8")).hexdigest()[:10]
        stem = raw_path.parent.name.replace("/", "_").replace(" ", "_")
        return f"{stem}_{digest}"

    def _infer_scenario_type(self, raw_path: Path) -> str:
        label_path = raw_path.parent / "scenario_label.json"
        if label_path.is_file():
            try:
                with open(label_path, "r", encoding="utf-8") as f:
                    payload = json.load(f)
                scenario_type = str(payload.get("scenario_type", "unknown")).strip()
                if scenario_type:
                    return scenario_type
            except Exception:
                pass

        # 兜底：尽量从父目录里猜，但最终保证有值
        for candidate in [raw_path.parent.name, raw_path.parent.parent.name if raw_path.parent.parent else ""]:
            text = str(candidate)
            if text in {"sudden_pedestrian_crossing", "cut_in", "hard_brake"}:
                return text
        return self.args.default_scenario_type

    def _write_scenario_type_gz(self, token_dir: Path, scenario_type: str) -> None:
        path = token_dir / "scenario_type.gz"
        payload = {"id": -1, "name": scenario_type}
        with gzip.open(path, "wb") as f:
            pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)

    def _copy_neighbor_metadata(self, raw_path: Path, token_dir: Path) -> None:
        for name in SIBLING_METADATA_FILES:
            src = raw_path.parent / name
            if src.is_file():
                shutil.copy2(src, token_dir / name)

    def _save_optional_preview(self, token_dir: Path, raster: Any) -> None:
        if not self.args.save_raster_npz:
            return
        raster_data = np.asarray(getattr(raster, "data", raster))
        np.savez_compressed(token_dir / "sledge_raster_preview.npz", raster=raster_data)

    def convert_one(self, raw_path: Path) -> Dict[str, Any]:
        raw_scene, source_format = load_raw_scene(raw_path)
        sledge_vector, sledge_raster = sledge_raw_feature_processing(raw_scene, self.sledge_config)
        if not isinstance(sledge_vector, SledgeVector):
            raise TypeError(f"sledge_raw_feature_processing 返回了非 SledgeVector 类型: {type(sledge_vector)}")

        scenario_type = self._infer_scenario_type(raw_path)
        token_name = self._token_name(raw_path)
        token_dir = self.output_root / "log" / scenario_type / token_name
        token_dir.mkdir(parents=True, exist_ok=True)

        self.feature_store.store_computed_feature_to_folder(token_dir / "sledge_vector", sledge_vector)
        self._write_scenario_type_gz(token_dir, scenario_type)
        self._copy_neighbor_metadata(raw_path, token_dir)
        self._save_optional_preview(token_dir, sledge_raster)

        conversion_meta = {
            "source_raw_path": str(raw_path),
            "source_format": source_format,
            "scenario_type": scenario_type,
            "token_dir": str(token_dir),
            "sledge_config": {f.name: _to_builtin(getattr(self.sledge_config, f.name)) for f in fields(SledgeConfig)},
            "vector_shapes": {
                "lines": list(np.asarray(sledge_vector.lines.states).shape),
                "vehicles": list(np.asarray(sledge_vector.vehicles.states).shape),
                "pedestrians": list(np.asarray(sledge_vector.pedestrians.states).shape),
                "static_objects": list(np.asarray(sledge_vector.static_objects.states).shape),
                "green_lights": list(np.asarray(sledge_vector.green_lights.states).shape),
                "red_lights": list(np.asarray(sledge_vector.red_lights.states).shape),
                "ego": list(np.asarray(sledge_vector.ego.states).shape),
            },
        }
        save_json(token_dir / "conversion_meta.json", conversion_meta)

        return {
            "source_raw_path": str(raw_path),
            "scenario_type": scenario_type,
            "token_dir": str(token_dir),
            "sledge_vector_path": str(resolve_gz_path(token_dir / "sledge_vector")),
            "converted": True,
        }

    def save_manifest(self) -> None:
        manifest_path = self.output_root / "conversion_manifest.csv"
        fieldnames = ["source_raw_path", "scenario_type", "token_dir", "sledge_vector_path", "converted"]
        with open(manifest_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row in self.rows:
                writer.writerow({k: row.get(k, "") for k in fieldnames})

        stats = {
            "total_found": len(self.raw_paths),
            "total_converted": len(self.rows),
            "total_failed": len(self.failed_rows),
            "scenario_counts": {},
            "output_root": str(self.output_root),
            "config": {f.name: _to_builtin(getattr(self.sledge_config, f.name)) for f in fields(SledgeConfig)},
        }
        for row in self.rows:
            scenario_type = str(row["scenario_type"])
            stats["scenario_counts"][scenario_type] = stats["scenario_counts"].get(scenario_type, 0) + 1

        save_json(self.output_root / "conversion_stats.json", stats)
        save_json(self.output_root / "conversion_failed_rows.json", self.failed_rows)

    def run(self) -> None:
        print(f"Found {len(self.raw_paths)} raw scene(s). output_root={self.output_root}")
        pbar = tqdm(self.raw_paths, total=len(self.raw_paths), dynamic_ncols=True, desc="raw_to_sim_vector_cache")
        for idx, raw_path in enumerate(pbar, start=1):
            try:
                row = self.convert_one(raw_path)
                self.rows.append(row)
                pbar.set_postfix(scene=row["scenario_type"], done=f"{idx}/{len(self.raw_paths)}")
            except Exception as exc:
                fail = {
                    "source_raw_path": str(raw_path),
                    "error_type": type(exc).__name__,
                    "error": repr(exc),
                }
                self.failed_rows.append(fail)
                pbar.set_postfix(failed=len(self.failed_rows), done=f"{idx}/{len(self.raw_paths)}")
                print(f"[FAIL] {raw_path}: {repr(exc)}")
        self.save_manifest()
        print(
            f"Finished. converted={len(self.rows)} failed={len(self.failed_rows)} output_root={self.output_root}"
        )



def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="把 semantic_control 生成的 sledge_raw.gz 批量转换为 simulation / NuBoard 可读取的 sledge_vector cache。"
    )
    parser.add_argument("--input", required=True, help="输入目录或单个 sledge_raw.gz 文件")
    parser.add_argument("--output-root", required=True, help="输出 cache 根目录")
    parser.add_argument("--config", default=None, help="可选：Hydra / YAML 配置文件，用于提取 SledgeConfig")
    parser.add_argument("--glob-pattern", default="**/sledge_raw.gz", help="目录模式，默认 **/sledge_raw.gz")
    parser.add_argument("--max-scenes", type=int, default=None, help="最多处理多少个场景")
    parser.add_argument(
        "--default-scenario-type",
        default="unknown",
        help="当找不到 scenario_label.json 时使用的场景名，默认 unknown",
    )
    parser.add_argument(
        "--save-raster-npz",
        action="store_true",
        help="额外保存一个 sledge_raster_preview.npz，便于离线排查",
    )
    return parser


if __name__ == "__main__":
    args = build_argparser().parse_args()
    RawToVectorCacheConverter(args).run()
