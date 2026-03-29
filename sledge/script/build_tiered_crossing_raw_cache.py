
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import pickle
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np
from omegaconf import OmegaConf

from sledge.semantic_control.io import load_raw_scene, save_raw_scene, save_json
from sledge.semantic_control.prompt_parser import NaturalLanguagePromptParser
from sledge.semantic_control.vector_editor import SemanticSceneEditor
from sledge.semantic_control.prompt_alignment import PromptAlignmentEvaluator
from sledge.autoencoder.preprocessing.features.sledge_vector_feature import SledgeVectorRaw


SEVERITY_TO_PROMPT = {
    "mild": "轻度 突发的行人横穿马路",
    "moderate": "中度 突发的行人横穿马路",
    "aggressive": "激进 突发的行人横穿马路",
}


def stable_bucket_from_path(path: Path) -> float:
    rel = str(path).encode("utf-8")
    digest = hashlib.md5(rel).hexdigest()
    value = int(digest[:12], 16)
    return value / float(16**12 - 1)


def choose_severity(bucket: float, mild_ratio: float, moderate_ratio: float, aggressive_ratio: float) -> str:
    total = mild_ratio + moderate_ratio + aggressive_ratio
    if total <= 0:
        raise ValueError("Ratios must sum to a positive number.")
    mild_cut = mild_ratio / total
    moderate_cut = (mild_ratio + moderate_ratio) / total
    if bucket < mild_cut:
        return "mild"
    if bucket < moderate_cut:
        return "moderate"
    return "aggressive"


def iter_scene_paths(input_dir: Path, glob_pattern: str = "**/sledge_raw.gz") -> List[Path]:
    return sorted(input_dir.glob(glob_pattern))


class TieredCrossingRawCacheBuilder:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.input_dir = Path(args.input_dir).resolve()
        self.output_root = Path(args.output_root).resolve()
        self.output_root.mkdir(parents=True, exist_ok=True)

        self.cfg = OmegaConf.load(args.config)
        self.prompt_parser = NaturalLanguagePromptParser()
        self.scene_editor = SemanticSceneEditor()
        self.alignment_evaluator = PromptAlignmentEvaluator()

        self.scene_paths = iter_scene_paths(self.input_dir, args.glob_pattern)
        if args.max_scenes is not None:
            self.scene_paths = self.scene_paths[: args.max_scenes]

        self.manifest_rows: List[Dict[str, Any]] = []

    def assign_severity(self, scene_path: Path) -> str:
        rel = scene_path.relative_to(self.input_dir)
        bucket = stable_bucket_from_path(rel)
        return choose_severity(bucket, self.args.mild_ratio, self.args.moderate_ratio, self.args.aggressive_ratio)

    def build_prompt(self, severity: str) -> str:
        if self.args.base_prompt:
            # prepend severity keyword to the user prompt
            prefix = {"mild": "轻度", "moderate": "中度", "aggressive": "激进"}[severity]
            return f"{prefix} {self.args.base_prompt}"
        return SEVERITY_TO_PROMPT[severity]

    def target_dir(self, scene_path: Path) -> Path:
        rel = scene_path.relative_to(self.input_dir)
        return self.output_root / rel.parent

    def run_one(self, idx: int, total: int, scene_path: Path) -> None:
        print(f"[{idx}/{total}] processing: {scene_path}")
        severity = self.assign_severity(scene_path)
        prompt = self.build_prompt(severity)
        out_dir = self.target_dir(scene_path)
        out_dir.mkdir(parents=True, exist_ok=True)

        marker = out_dir / "severity_label.json"
        if self.args.skip_existing and marker.exists() and (out_dir / "sledge_raw.gz").exists():
            print(f"[{idx}/{total}] skipped: {scene_path}")
            return

        raw_scene, source_format = load_raw_scene(scene_path)
        prompt_spec = self.prompt_parser.parse(prompt)
        edited_raw, edit_report = self.scene_editor.edit(raw_scene, prompt_spec)
        alignment = self.alignment_evaluator.evaluate(edited_raw, prompt_spec)

        save_raw_scene(out_dir / "sledge_raw", edited_raw, source_format=source_format)
        save_json(out_dir / "severity_label.json", {"severity_level": severity, "prompt": prompt})
        save_json(out_dir / "edit_report.json", edit_report)
        save_json(out_dir / "edited_prompt_alignment.json", alignment.to_dict())

        summary = {
            "scene_path": str(scene_path),
            "output_dir": str(out_dir),
            "severity_level": severity,
            "prompt": prompt,
            "source_format": source_format,
            "edited_alignment": float(alignment.total),
            "accepted": bool(getattr(alignment, "accepted", False)),
        }
        save_json(out_dir / "summary.json", summary)

        self.manifest_rows.append(
            {
                "scene_path": str(scene_path),
                "relative_scene_dir": str(scene_path.relative_to(self.input_dir).parent),
                "output_dir": str(out_dir),
                "severity_level": severity,
                "prompt": prompt,
                "edited_alignment": float(alignment.total),
                "accepted": bool(getattr(alignment, "accepted", False)),
            }
        )
        print(
            f"[{idx}/{total}] done: {scene_path} | severity={severity} | edited_alignment={float(alignment.total):.4f}"
        )

    def save_manifest(self) -> None:
        manifest_path = self.output_root / "severity_manifest.csv"
        fieldnames = [
            "scene_path",
            "relative_scene_dir",
            "output_dir",
            "severity_level",
            "prompt",
            "edited_alignment",
            "accepted",
        ]
        with open(manifest_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row in self.manifest_rows:
                writer.writerow(row)

        counts = {"mild": 0, "moderate": 0, "aggressive": 0}
        for row in self.manifest_rows:
            counts[row["severity_level"]] += 1
        save_json(
            self.output_root / "severity_stats.json",
            {
                "counts": counts,
                "ratios": {
                    "mild": self.args.mild_ratio,
                    "moderate": self.args.moderate_ratio,
                    "aggressive": self.args.aggressive_ratio,
                },
                "total_processed": len(self.manifest_rows),
            },
        )

    def run_batch(self) -> None:
        total = len(self.scene_paths)
        print(f"Found {total} scene(s). output_root={self.output_root}")
        for idx, scene_path in enumerate(self.scene_paths, start=1):
            try:
                self.run_one(idx, total, scene_path)
            except Exception as e:
                print(f"[{idx}/{total}] failed: {scene_path}")
                print(repr(e))
        self.save_manifest()


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--base-prompt", default="突发的行人横穿马路")
    parser.add_argument("--glob-pattern", default="**/sledge_raw.gz")
    parser.add_argument("--max-scenes", type=int, default=None)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--mild-ratio", type=float, default=0.50)
    parser.add_argument("--moderate-ratio", type=float, default=0.35)
    parser.add_argument("--aggressive-ratio", type=float, default=0.15)
    return parser


def main() -> None:
    parser = build_argparser()
    args = parser.parse_args()
    builder = TieredCrossingRawCacheBuilder(args)
    builder.run_batch()


if __name__ == "__main__":
    main()
