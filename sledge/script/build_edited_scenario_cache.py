from __future__ import annotations

import argparse
import json
import os
import traceback
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np
from omegaconf import OmegaConf

from sledge.autoencoder.modeling.models.rvae.rvae_config import RVAEConfig
from sledge.autoencoder.preprocessing.feature_builders.sledge.sledge_feature_processing import (
    sledge_raw_feature_processing,
)
from sledge.autoencoder.preprocessing.features.map_id_feature import MAP_ID_TO_NAME
from sledge.autoencoder.preprocessing.features.sledge_raster_feature import SledgeRaster
from sledge.autoencoder.preprocessing.features.sledge_vector_feature import (
    SledgeVector,
    SledgeVectorElement,
)
from sledge.common.visualization.sledge_visualization_utils import (
    get_sledge_raster,
    get_sledge_vector_as_raster,
)
from sledge.semantic_control import (
    NaturalLanguagePromptParser,
    PromptAlignmentEvaluator,
    SemanticSceneEditor,
)
from sledge.semantic_control.io import (
    feature_to_raw_scene_dict,
    load_raw_scene,
    save_gz_pickle,
    save_json,
    save_raw_scene,
)


DEFAULT_ALIGNMENT_THRESHOLD = 0.70


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build an edited-only scenario cache for the simplified 'sudden pedestrian crossing' semantics, without running diffusion."
    )
    source_group = parser.add_mutually_exclusive_group(required=True)
    source_group.add_argument("--input", help="Path to one source sledge_raw.gz")
    source_group.add_argument("--input-dir", help="Cache root; recursively process all matching sledge_raw.gz files")

    parser.add_argument("--output", required=True, help="Output directory for reports / debug artifacts")
    parser.add_argument("--prompt", required=True, help="Natural-language control prompt")
    parser.add_argument("--config", required=True, help="Expanded OmegaConf yaml containing autoencoder_model.config")
    parser.add_argument(
        "--scenario-cache-root",
        default=None,
        help="Where to write final simulation-readable sledge_vector.gz. Defaults to $SLEDGE_EXP_ROOT/caches/scenario_cache_edited_only",
    )
    parser.add_argument("--map-id", type=int, default=None, help="Optional override for city label")
    parser.add_argument("--glob-pattern", default="**/sledge_raw.gz")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--max-scenes", type=int, default=None)
    parser.add_argument("--output-layout", choices=["mirror", "flat"], default="mirror")
    parser.add_argument("--alignment-threshold", type=float, default=DEFAULT_ALIGNMENT_THRESHOLD)
    parser.add_argument(
        "--save-visuals",
        action="store_true",
        help="Try to save edited_raster.png / edited_vector.png. Visualization failures will not stop the main pipeline.",
    )
    return parser


def resolve_map_id(scene_path: Path, override_map_id: Optional[int], parsed_map_id: Optional[int]) -> int:
    if override_map_id is not None:
        return int(override_map_id)
    if parsed_map_id is not None:
        return int(parsed_map_id)
    path_str = str(scene_path).lower()
    for map_id, map_name in MAP_ID_TO_NAME.items():
        if map_name in path_str:
            return int(map_id)
    return 3


def save_image(path: Path, image_rgb: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    cv2.imwrite(str(path), image_bgr)


_SEVERITY_ACCEPTANCE = {
    "mild": {
        "pedestrian_presence_score": 0.75,
        "roadside_emergence_score": 0.25,
        "crossing_direction_score": 0.35,
        "ego_lane_conflict_score": 0.30,
        "immediacy_score": 0.08,
        "total": 0.70,
    },
    "moderate": {
        "pedestrian_presence_score": 0.75,
        "roadside_emergence_score": 0.30,
        "crossing_direction_score": 0.40,
        "ego_lane_conflict_score": 0.35,
        "immediacy_score": 0.15,
        "total": 0.72,
    },
    "aggressive": {
        "pedestrian_presence_score": 0.80,
        "roadside_emergence_score": 0.30,
        "crossing_direction_score": 0.45,
        "ego_lane_conflict_score": 0.40,
        "immediacy_score": 0.25,
        "total": 0.75,
    },
}


def summarize_crossing_semantics(
    alignment: object,
    prompt_spec: object,
    threshold: float,
) -> Dict[str, object]:
    details = dict(getattr(alignment, "details", {}) or {})
    notes = list(getattr(alignment, "notes", []) or [])
    total = float(getattr(alignment, "total", 0.0))
    scenario_type = str(getattr(prompt_spec, "scenario_type", "generic"))
    severity_level = str(getattr(prompt_spec, "severity_level", "moderate") or "moderate").lower()
    if severity_level not in _SEVERITY_ACCEPTANCE:
        severity_level = "moderate"
    crossing_prompt = scenario_type in {"pedestrian_crossing", "sudden_pedestrian_crossing"}

    ped = float(details.get("pedestrian_presence_score", 0.0))
    roadside = float(details.get("roadside_emergence_score", 0.0))
    direction = float(details.get("crossing_direction_score", 0.0))
    conflict = float(details.get("ego_lane_conflict_score", 0.0))
    immediacy = float(details.get("immediacy_score", 0.0))

    tier_thresholds = dict(_SEVERITY_ACCEPTANCE[severity_level])
    tier_thresholds["total"] = max(float(tier_thresholds["total"]), float(threshold))

    checks = {
        "pedestrian_presence_ok": ped >= tier_thresholds["pedestrian_presence_score"],
        "roadside_emergence_ok": roadside >= tier_thresholds["roadside_emergence_score"],
        "crossing_direction_ok": direction >= tier_thresholds["crossing_direction_score"],
        "ego_lane_conflict_ok": conflict >= tier_thresholds["ego_lane_conflict_score"],
        "immediacy_ok": immediacy >= tier_thresholds["immediacy_score"],
        "total_ok": total >= tier_thresholds["total"],
    }

    if crossing_prompt:
        semantic_pass = all(checks.values())
    else:
        semantic_pass = total >= float(threshold)

    return {
        "crossing_prompt": crossing_prompt,
        "scenario_type": scenario_type,
        "severity_level": severity_level,
        "pedestrian_presence_score": ped,
        "roadside_emergence_score": roadside,
        "crossing_direction_score": direction,
        "ego_lane_conflict_score": conflict,
        "immediacy_score": immediacy,
        "semantic_pass": bool(semantic_pass),
        "threshold": float(threshold),
        "effective_thresholds": tier_thresholds,
        "checks": checks,
        "notes": notes,
    }


class EditedScenarioCacheBuilder:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.out_root = Path(args.output)
        self.out_root.mkdir(parents=True, exist_ok=True)

        self.cfg = OmegaConf.load(args.config)
        ae_cfg_dict = OmegaConf.to_container(self.cfg.autoencoder_model.config, resolve=True)
        if not isinstance(ae_cfg_dict, dict):
            raise TypeError(f"Expected autoencoder_model.config to resolve to dict, got {type(ae_cfg_dict)}")
        filtered = {k: v for k, v in ae_cfg_dict.items() if k in RVAEConfig.__annotations__}
        self.ae_config = RVAEConfig(**filtered)

        self.prompt_parser = NaturalLanguagePromptParser()
        self.scene_editor = SemanticSceneEditor()
        self.alignment_evaluator = PromptAlignmentEvaluator()
        self.scenario_cache_root = self._resolve_scenario_cache_root(args.scenario_cache_root)
        self.scenario_cache_root.mkdir(parents=True, exist_ok=True)

    def _resolve_scenario_cache_root(self, override: Optional[str]) -> Path:
        if override:
            return Path(override)
        sledge_exp_root = os.environ.get("SLEDGE_EXP_ROOT")
        if not sledge_exp_root:
            raise EnvironmentError(
                "SLEDGE_EXP_ROOT is not set. Export it or pass --scenario-cache-root explicitly."
            )
        return Path(sledge_exp_root) / "caches" / "scenario_cache_edited_only"

    def _scene_output_dir(self, scene_path: Path, index: int) -> Path:
        if self.args.input:
            return self.out_root
        assert self.args.input_dir is not None
        root = Path(self.args.input_dir)
        if self.args.output_layout == "flat":
            stem = scene_path.parent.name
            return self.out_root / f"{index:06d}_{stem}"
        rel = scene_path.parent.relative_to(root)
        return self.out_root / rel

    def _scenario_cache_dir(self, scene_path: Path, index: int) -> Path:
        if self.args.input_dir:
            rel = scene_path.parent.relative_to(Path(self.args.input_dir))
            return self.scenario_cache_root / rel

        parts = scene_path.parent.parts
        if len(parts) >= 3:
            rel = Path(*parts[-3:])
        else:
            rel = Path(scene_path.parent.name)
        if self.args.output_layout == "flat":
            rel = Path(f"{index:06d}_{scene_path.parent.name}")
        return self.scenario_cache_root / rel

    @staticmethod
    def make_simulation_compatible_vector(processed_vector: SledgeVector, edited_raw) -> SledgeVector:
        raw_ego_states = np.asarray(edited_raw.ego.states)
        raw_ego_mask = np.asarray(edited_raw.ego.mask)
        ego_speed = float(raw_ego_states[0]) if raw_ego_states.size > 0 else 0.0
        ego_valid = bool(raw_ego_mask.reshape(-1)[0]) if raw_ego_mask.size > 0 else True
        sim_ego = SledgeVectorElement(
            states=np.asarray([ego_speed], dtype=np.float32),
            mask=np.asarray([ego_valid], dtype=np.float32),
        )
        return SledgeVector(
            lines=processed_vector.lines,
            vehicles=processed_vector.vehicles,
            pedestrians=processed_vector.pedestrians,
            static_objects=processed_vector.static_objects,
            green_lights=processed_vector.green_lights,
            red_lights=processed_vector.red_lights,
            ego=sim_ego,
        )

    def _maybe_save_visuals(self, out_dir: Path, processed_vector: SledgeVector, processed_raster: SledgeRaster) -> Optional[Dict[str, object]]:
        if not self.args.save_visuals:
            return None
        try:
            visual_raster = processed_raster.to_feature_tensor()
            visual_raster = SledgeRaster(visual_raster.data.unsqueeze(0).cpu())
            save_image(out_dir / "edited_raster.png", get_sledge_raster(visual_raster, self.ae_config.pixel_frame))
            save_image(out_dir / "edited_vector.png", get_sledge_vector_as_raster(processed_vector, self.ae_config))
            return None
        except Exception as exc:
            warning = {
                "warning_type": type(exc).__name__,
                "warning": repr(exc),
                "traceback": traceback.format_exc(),
            }
            save_json(out_dir / "visualization_warning.json", warning)
            return warning

    def run_one(self, scene_path: Path, out_dir: Path, index: int = 1) -> Dict[str, object]:
        out_dir.mkdir(parents=True, exist_ok=True)
        raw_scene, source_format = load_raw_scene(scene_path)
        prompt_spec = self.prompt_parser.parse(self.args.prompt)
        map_id = resolve_map_id(scene_path, self.args.map_id, prompt_spec.map_id)

        edited_raw, edit_result = self.scene_editor.edit(raw_scene, prompt_spec)
        processed_vector, processed_raster = sledge_raw_feature_processing(edited_raw, self.ae_config)
        alignment = self.alignment_evaluator.evaluate(processed_vector, prompt_spec)
        semantic_summary = summarize_crossing_semantics(alignment, prompt_spec, self.args.alignment_threshold)

        save_raw_scene(out_dir / "edited_sledge_raw", edited_raw, source_format=source_format)
        save_json(out_dir / "prompt_spec.json", prompt_spec.to_dict())
        save_json(out_dir / "edit_report.json", edit_result.to_dict())
        save_json(
            out_dir / "edited_prompt_alignment.json",
            {
                **alignment.to_dict(),
                **semantic_summary,
                "accepted": bool(semantic_summary["semantic_pass"]),
                "map_id": int(map_id),
                "map_name": MAP_ID_TO_NAME.get(map_id, "unknown"),
                "source_format": source_format,
            },
        )

        visualization_warning = self._maybe_save_visuals(out_dir, processed_vector, processed_raster)

        sim_vector = self.make_simulation_compatible_vector(processed_vector, edited_raw)
        scenario_cache_dir = self._scenario_cache_dir(scene_path, index)
        scenario_cache_dir.mkdir(parents=True, exist_ok=True)
        scenario_vector_path = save_gz_pickle(
            scenario_cache_dir / "sledge_vector",
            feature_to_raw_scene_dict(sim_vector),
        )

        summary = {
            "scene_path": str(scene_path),
            "output_dir": str(out_dir),
            "scenario_cache_vector_path": str(scenario_vector_path),
            "map_id": int(map_id),
            "map_name": MAP_ID_TO_NAME.get(map_id, "unknown"),
            "source_format": source_format,
            "prompt_type": prompt_spec.scenario_type,
            "edited_alignment": float(alignment.total),
            "accepted": bool(semantic_summary["semantic_pass"]),
            "semantic_summary": semantic_summary,
            "visualization_warning": visualization_warning is not None,
        }
        save_json(out_dir / "summary.json", summary)
        return summary

    def iter_scene_paths(self) -> List[Path]:
        if self.args.input:
            return [Path(self.args.input)]
        scene_paths = sorted(Path(self.args.input_dir).glob(self.args.glob_pattern))
        if self.args.max_scenes is not None:
            scene_paths = scene_paths[: self.args.max_scenes]
        return scene_paths

    def run_batch(self) -> None:
        scene_paths = self.iter_scene_paths()
        total = len(scene_paths)
        print(f"Found {total} scene(s). scenario_cache_root={self.scenario_cache_root}")
        summary_rows: List[Dict[str, object]] = []

        for index, scene_path in enumerate(scene_paths, start=1):
            out_dir = self._scene_output_dir(scene_path, index)
            scenario_cache_dir = self._scenario_cache_dir(scene_path, index)
            marker = out_dir / "edited_prompt_alignment.json"
            scenario_marker = scenario_cache_dir / "sledge_vector.gz"
            if self.args.skip_existing and marker.exists() and scenario_marker.exists():
                print(f"[{index}/{total}] skipped: {scene_path}")
                continue

            print(f"[{index}/{total}] processing: {scene_path}")
            try:
                row = self.run_one(scene_path, out_dir, index=index)
                summary_rows.append(row)
                print(
                    f"[{index}/{total}] done: {scene_path} | edited_alignment={row['edited_alignment']:.4f} | accepted={row['accepted']} | scenario_cache={row['scenario_cache_vector_path']}"
                )
            except Exception as exc:
                error_payload = {
                    "scene_path": str(scene_path),
                    "error_type": type(exc).__name__,
                    "error": repr(exc),
                    "traceback": traceback.format_exc(),
                }
                out_dir.mkdir(parents=True, exist_ok=True)
                save_json(out_dir / "error.json", error_payload)
                print(f"[{index}/{total}] failed: {scene_path}\n{repr(exc)}")

        batch_summary = {
            "total_seen": total,
            "finished": len(summary_rows),
            "accepted": int(sum(bool(row["accepted"]) for row in summary_rows)),
            "scenario_cache_root": str(self.scenario_cache_root),
            "rows": summary_rows,
        }
        save_json(self.out_root / "batch_summary.json", batch_summary)
        with open(self.out_root / "batch_summary.jsonl", "w", encoding="utf-8") as fp:
            for row in summary_rows:
                fp.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    args = build_argparser().parse_args()
    builder = EditedScenarioCacheBuilder(args)
    if args.input:
        builder.run_one(Path(args.input), Path(args.output), index=1)
        return
    builder.run_batch()


if __name__ == "__main__":
    main()
