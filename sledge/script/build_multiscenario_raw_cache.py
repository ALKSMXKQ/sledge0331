from __future__ import annotations

import argparse
import csv
import hashlib
import traceback
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List

from omegaconf import OmegaConf
from tqdm import tqdm

from sledge.semantic_control.io import load_raw_scene, save_json, save_raw_scene
from sledge.semantic_control.prompt_alignment import PromptAlignmentEvaluator
from sledge.semantic_control.prompt_parser import NaturalLanguagePromptParser
from sledge.semantic_control.vector_editor import SemanticSceneEditor


SCENARIO_TO_BASE_PROMPT = {
    "sudden_pedestrian_crossing": "突发的行人横穿马路",
    "cut_in": "邻车突然加塞到自车前方",
    "hard_brake": "自车前方车辆突然急刹",
}

SEVERITY_PREFIX = {
    "mild": "轻度",
    "moderate": "中度",
    "aggressive": "激进",
}


def stable_bucket_from_text(text: str) -> float:
    digest = hashlib.md5(text.encode("utf-8")).hexdigest()
    value = int(digest[:12], 16)
    return value / float(16**12 - 1)


def choose_from_ratios(bucket: float, items: Dict[str, float]) -> str:
    total = sum(max(v, 0.0) for v in items.values())
    if total <= 0:
        raise ValueError("Ratios must sum to a positive number.")
    running = 0.0
    for key, value in items.items():
        running += max(value, 0.0) / total
        if bucket < running:
            return key
    return list(items.keys())[-1]


def iter_scene_paths(input_dir: Path, glob_pattern: str = "**/sledge_raw.gz") -> List[Path]:
    return sorted(input_dir.glob(glob_pattern))


class MultiScenarioRawCacheBuilder:
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
        self.failed_rows: List[Dict[str, Any]] = []

    def assign_scenario_type(self, scene_path: Path) -> str:
        rel = str(scene_path.relative_to(self.input_dir))
        bucket = stable_bucket_from_text("scenario::" + rel)
        return choose_from_ratios(
            bucket,
            {
                "sudden_pedestrian_crossing": self.args.crossing_ratio,
                "cut_in": self.args.cut_in_ratio,
                "hard_brake": self.args.hard_brake_ratio,
            },
        )

    def assign_severity(self, scene_path: Path) -> str:
        rel = str(scene_path.relative_to(self.input_dir))
        bucket = stable_bucket_from_text("severity::" + rel)
        return choose_from_ratios(
            bucket,
            {
                "mild": self.args.mild_ratio,
                "moderate": self.args.moderate_ratio,
                "aggressive": self.args.aggressive_ratio,
            },
        )

    def build_prompt(self, scenario_type: str, severity: str) -> str:
        if self.args.base_prompt:
            return f"{SEVERITY_PREFIX[severity]} {self.args.base_prompt}"
        return f"{SEVERITY_PREFIX[severity]} {SCENARIO_TO_BASE_PROMPT[scenario_type]}"

    def target_dir(self, scene_path: Path) -> Path:
        rel = scene_path.relative_to(self.input_dir)
        return self.output_root / rel.parent

    def run_one(self, scene_path: Path) -> Dict[str, Any]:
        scenario_type = self.assign_scenario_type(scene_path)
        severity = self.assign_severity(scene_path)
        prompt = self.build_prompt(scenario_type, severity)

        out_dir = self.target_dir(scene_path)
        out_dir.mkdir(parents=True, exist_ok=True)

        marker = out_dir / "scenario_label.json"
        if self.args.skip_existing and marker.exists() and (out_dir / "sledge_raw.gz").exists():
            return {
                "skipped": True,
                "scenario_type": scenario_type,
                "severity_level": severity,
                "output_dir": str(out_dir),
            }

        raw_scene, source_format = load_raw_scene(scene_path)
        prompt_spec = self.prompt_parser.parse(prompt)
        edited_raw, edit_report = self.scene_editor.edit(raw_scene, prompt_spec)
        alignment = self.alignment_evaluator.evaluate(edited_raw, prompt_spec)

        alignment_dict = alignment.to_dict()
        primary_note = alignment_dict["notes"][0] if alignment_dict.get("notes") else ""

        save_raw_scene(out_dir / "sledge_raw", edited_raw, source_format=source_format)
        save_json(
            out_dir / "scenario_label.json",
            {
                "scenario_type": scenario_type,
                "severity_level": severity,
                "prompt": prompt,
                "source_scene_path": str(scene_path),
                "source_format": source_format,
                "edited_alignment": float(alignment.total),
                "accepted": bool(getattr(alignment, "accepted", False)),
                "primary_note": primary_note,
            },
        )
        save_json(out_dir / "edit_report.json", edit_report.to_dict() if hasattr(edit_report, "to_dict") else edit_report)
        save_json(out_dir / "edited_prompt_alignment.json", alignment_dict)

        summary = {
            "scene_path": str(scene_path),
            "output_dir": str(out_dir),
            "scenario_type": scenario_type,
            "severity_level": severity,
            "prompt": prompt,
            "source_format": source_format,
            "edited_alignment": float(alignment.total),
            "accepted": bool(getattr(alignment, "accepted", False)),
            "primary_note": primary_note,
        }
        save_json(out_dir / "summary.json", summary)
        return summary

    def save_manifest(self) -> None:
        manifest_path = self.output_root / "scenario_manifest.csv"
        fieldnames = [
            "scene_path",
            "output_dir",
            "scenario_type",
            "severity_level",
            "prompt",
            "edited_alignment",
            "accepted",
            "primary_note",
        ]
        with open(manifest_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row in self.manifest_rows:
                writer.writerow({k: row.get(k, "") for k in fieldnames})

        accepted_manifest_path = self.output_root / "accepted_manifest.csv"
        with open(accepted_manifest_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row in self.manifest_rows:
                if bool(row.get("accepted", False)):
                    writer.writerow({k: row.get(k, "") for k in fieldnames})

        stats = {
            "total_processed": len(self.manifest_rows),
            "total_failed": len(self.failed_rows),
            "scenario_counts": {},
            "severity_counts": {},
            "accepted_count": int(sum(bool(r.get("accepted", False)) for r in self.manifest_rows)),
            "mean_alignment": (
                sum(float(r.get("edited_alignment", 0.0)) for r in self.manifest_rows) / max(1, len(self.manifest_rows))
            ),
            "ratios": {
                "crossing_ratio": self.args.crossing_ratio,
                "cut_in_ratio": self.args.cut_in_ratio,
                "hard_brake_ratio": self.args.hard_brake_ratio,
                "mild_ratio": self.args.mild_ratio,
                "moderate_ratio": self.args.moderate_ratio,
                "aggressive_ratio": self.args.aggressive_ratio,
            },
            "scenario_acceptance": {},
            "severity_acceptance": {},
            "scenario_severity_counts": {},
            "scenario_severity_acceptance": {},
            "top_rejection_notes": [],
            "score_buckets": {},
        }

        scenario_counter: Counter[str] = Counter()
        scenario_accept_counter: Counter[str] = Counter()
        severity_counter: Counter[str] = Counter()
        severity_accept_counter: Counter[str] = Counter()
        scenario_sev_counter: Dict[str, Counter[str]] = defaultdict(Counter)
        scenario_sev_accept_counter: Dict[str, Counter[str]] = defaultdict(Counter)
        reject_note_counter: Counter[str] = Counter()
        score_buckets = {
            "[0.00,0.20)": 0,
            "[0.20,0.50)": 0,
            "[0.50,0.70)": 0,
            "[0.70,0.90)": 0,
            "[0.90,1.00]": 0,
        }

        for row in self.manifest_rows:
            s = row["scenario_type"]
            sev = row["severity_level"]
            acc = bool(row.get("accepted", False))
            score = float(row.get("edited_alignment", 0.0))

            scenario_counter[s] += 1
            severity_counter[sev] += 1
            scenario_sev_counter[s][sev] += 1
            if acc:
                scenario_accept_counter[s] += 1
                severity_accept_counter[sev] += 1
                scenario_sev_accept_counter[s][sev] += 1
            else:
                note = str(row.get("primary_note", "")).strip() or "no_note"
                reject_note_counter[note] += 1

            if score < 0.20:
                score_buckets["[0.00,0.20)"] += 1
            elif score < 0.50:
                score_buckets["[0.20,0.50)"] += 1
            elif score < 0.70:
                score_buckets["[0.50,0.70)"] += 1
            elif score < 0.90:
                score_buckets["[0.70,0.90)"] += 1
            else:
                score_buckets["[0.90,1.00]"] += 1

        stats["scenario_counts"] = dict(scenario_counter)
        stats["severity_counts"] = dict(severity_counter)
        stats["score_buckets"] = score_buckets
        stats["top_rejection_notes"] = [
            {"note": note, "count": int(count)} for note, count in reject_note_counter.most_common(10)
        ]

        for s, count in scenario_counter.items():
            stats["scenario_acceptance"][s] = {
                "accepted": int(scenario_accept_counter[s]),
                "total": int(count),
                "accept_rate": float(scenario_accept_counter[s] / max(1, count)),
            }

        for sev, count in severity_counter.items():
            stats["severity_acceptance"][sev] = {
                "accepted": int(severity_accept_counter[sev]),
                "total": int(count),
                "accept_rate": float(severity_accept_counter[sev] / max(1, count)),
            }

        for s, counter in scenario_sev_counter.items():
            stats["scenario_severity_counts"][s] = dict(counter)
            stats["scenario_severity_acceptance"][s] = {}
            for sev, count in counter.items():
                accepted = int(scenario_sev_accept_counter[s][sev])
                stats["scenario_severity_acceptance"][s][sev] = {
                    "accepted": accepted,
                    "total": int(count),
                    "accept_rate": float(accepted / max(1, count)),
                }

        save_json(self.output_root / "scenario_stats.json", stats)
        save_json(self.output_root / "failed_rows.json", self.failed_rows)

    def run_batch(self) -> None:
        total = len(self.scene_paths)
        print(f"Found {total} scene(s). output_root={self.output_root}")

        pbar = tqdm(self.scene_paths, total=total, dynamic_ncols=True, desc="build_multiscenario_raw_cache")
        for idx, scene_path in enumerate(pbar, start=1):
            try:
                row = self.run_one(scene_path)
                if row.get("skipped", False):
                    pbar.set_postfix(
                        scene=row.get("scenario_type", "unknown"),
                        severity=row.get("severity_level", "unknown"),
                        skipped=idx,
                    )
                    continue

                self.manifest_rows.append(row)
                pbar.set_postfix(
                    scene=row["scenario_type"],
                    severity=row["severity_level"],
                    align=f'{float(row["edited_alignment"]):.3f}',
                    accepted=bool(row["accepted"]),
                    done=f"{idx}/{total}",
                )
                pbar.write(
                    f"[{idx}/{total}] done: {scene_path} | "
                    f"scenario={row['scenario_type']} | severity={row['severity_level']} | "
                    f"edited_alignment={float(row['edited_alignment']):.4f} | accepted={bool(row['accepted'])}"
                )
            except Exception as exc:
                error_payload = {
                    "scene_path": str(scene_path),
                    "error_type": type(exc).__name__,
                    "error": repr(exc),
                    "traceback": traceback.format_exc(),
                }
                self.failed_rows.append(error_payload)
                fail_dir = self.target_dir(scene_path)
                fail_dir.mkdir(parents=True, exist_ok=True)
                save_json(fail_dir / "error.json", error_payload)
                pbar.write(f"[{idx}/{total}] failed: {scene_path}\n{repr(exc)}")
                pbar.set_postfix(failed=len(self.failed_rows), done=f"{idx}/{total}")

        self.save_manifest()
        accepted_count = int(sum(bool(r.get("accepted", False)) for r in self.manifest_rows))
        print(
            f"Finished. processed={len(self.manifest_rows)} failed={len(self.failed_rows)} "
            f"accepted={accepted_count} output_root={self.output_root}"
        )


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--base-prompt", default="")
    parser.add_argument("--glob-pattern", default="**/sledge_raw.gz")
    parser.add_argument("--max-scenes", type=int, default=None)
    parser.add_argument("--skip-existing", action="store_true")

    parser.add_argument("--crossing-ratio", type=float, default=0.40)
    parser.add_argument("--cut-in-ratio", type=float, default=0.30)
    parser.add_argument("--hard-brake-ratio", type=float, default=0.30)

    parser.add_argument("--mild-ratio", type=float, default=0.50)
    parser.add_argument("--moderate-ratio", type=float, default=0.35)
    parser.add_argument("--aggressive-ratio", type=float, default=0.15)
    return parser


def main() -> None:
    args = build_argparser().parse_args()
    builder = MultiScenarioRawCacheBuilder(args)
    builder.run_batch()


if __name__ == "__main__":
    main()
