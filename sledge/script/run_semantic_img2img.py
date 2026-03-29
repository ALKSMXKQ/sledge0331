
from __future__ import annotations

import argparse
import json
import math
import os
import traceback
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf

from sledge.autoencoder.modeling.models.rvae.rvae_config import RVAEConfig
from sledge.autoencoder.preprocessing.feature_builders.sledge.sledge_feature_processing import (
    sledge_raw_feature_processing,
)
from sledge.autoencoder.preprocessing.features.map_id_feature import MAP_ID_TO_NAME
from sledge.autoencoder.preprocessing.features.sledge_raster_feature import SledgeRaster
from sledge.autoencoder.preprocessing.features.sledge_vector_feature import (
    AgentIndex,
    StaticObjectIndex,
    SledgeConfig,
    SledgeVector,
    SledgeVectorElement,
)
from sledge.common.visualization.sledge_visualization_utils import (
    get_sledge_raster,
    get_sledge_vector_as_raster,
)
from sledge.script.builders.diffusion_builder import build_pipeline_from_checkpoint
from sledge.script.builders.model_builder import build_autoencoder_torch_module_wrapper
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
from sledge.semantic_control.prompt_spec import SceneEditROI


FORWARD_X_MIN = 5.0
FORWARD_X_MAX = 22.0
CORRIDOR_HALF_WIDTH = 2.6
STRICT_MIN_FORWARD_CORRIDOR = 0.70
STRICT_MIN_BLIND_SPOT = 0.55


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Semantic scene editing + low-noise masked img2img denoising with strict forward ghost-probe preservation"
    )
    source_group = parser.add_mutually_exclusive_group(required=True)
    source_group.add_argument("--input", help="Path to one source sledge_raw.gz")
    source_group.add_argument(
        "--input-dir",
        help="Cache root; recursively process all matching sledge_raw.gz files",
    )

    parser.add_argument("--output", required=True, help="Output directory for reports / debug artifacts")
    parser.add_argument("--prompt", required=True, help="Natural-language control prompt")
    parser.add_argument("--config", required=True, help="Expanded OmegaConf yaml")
    parser.add_argument("--autoencoder-checkpoint", required=True, help="RVAE checkpoint path")
    parser.add_argument("--diffusion-checkpoint", required=True, help="DiT / diffusion pipeline checkpoint path")
    parser.add_argument(
        "--scenario-cache-root",
        default=None,
        help="Where to write final sledge_vector.gz. Defaults to $SLEDGE_EXP_ROOT/caches/scenario_cache",
    )
    parser.add_argument("--map-id", type=int, default=None, help="Optional override for city label")

    # More conservative defaults than before: lower corruption, stronger preservation.
    parser.add_argument("--num-inference-timesteps", type=int, default=24)
    parser.add_argument("--start-timestep-index", type=int, default=12)
    parser.add_argument("--guidance-scale", type=float, default=3.0)
    parser.add_argument("--mask-dilation", type=int, default=3, help="Base latent-cell dilation around preserved ROI")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--save-latents", action="store_true")
    parser.add_argument("--alignment-threshold", type=float, default=0.8)
    parser.add_argument("--glob-pattern", default="**/sledge_raw.gz")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--max-scenes", type=int, default=None)
    parser.add_argument("--output-layout", choices=["mirror", "flat"], default="mirror")

    # Strong generation constraints.
    parser.add_argument("--resample-attempts", type=int, default=3)
    parser.add_argument("--strict-min-forward-corridor", type=float, default=STRICT_MIN_FORWARD_CORRIDOR)
    parser.add_argument("--strict-min-blind-spot", type=float, default=STRICT_MIN_BLIND_SPOT)
    parser.add_argument("--strict-forward-x-min", type=float, default=FORWARD_X_MIN)
    parser.add_argument("--strict-forward-x-max", type=float, default=FORWARD_X_MAX)
    parser.add_argument("--strict-corridor-half-width", type=float, default=CORRIDOR_HALF_WIDTH)
    parser.add_argument("--allow-write-failed-scenario-cache", action="store_true")

    # Corridor-preservation controls.
    parser.add_argument("--pedestrian-future-horizon-s", type=float, default=2.0)
    parser.add_argument("--preserve-corridor-band-half-width", type=float, default=1.6)
    parser.add_argument("--preserve-occ-ped-bridge-extra-width", type=float, default=1.0)
    parser.add_argument("--low-noise-start-step-seq", default="8,10,12", help="Comma-separated start-step candidates")
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


def encode_raster(autoencoder_model, raster: SledgeRaster, device: str) -> torch.Tensor:
    raster_tensor = raster.to_feature_tensor().data.unsqueeze(0).to(device)
    encoder = autoencoder_model.get_encoder().to(device)
    encoder.eval()
    with torch.no_grad():
        latent_dist = encoder(raster_tensor)
    return latent_dist.mu


def _valid_states(elem: SledgeVectorElement) -> np.ndarray:
    valid = np.asarray(elem.mask).astype(bool)
    return elem.states[valid] if np.any(valid) else np.zeros((0, elem.states.shape[-1]), dtype=np.asarray(elem.states).dtype)


def _interval_overlap_ratio(a0: float, a1: float, b0: float, b1: float) -> float:
    inter = max(0.0, min(a1, b1) - max(a0, b0))
    denom = max(1e-6, min(a1 - a0, b1 - b0))
    return inter / denom


def _angular_interval_from_bbox(center_xy: np.ndarray, length: float, width: float) -> Tuple[float, float]:
    cx, cy = float(center_xy[0]), float(center_xy[1])
    half_l = max(length * 0.5, 0.2)
    half_w = max(width * 0.5, 0.2)
    corners = np.array(
        [
            [cx - half_l, cy - half_w],
            [cx - half_l, cy + half_w],
            [cx + half_l, cy - half_w],
            [cx + half_l, cy + half_w],
        ],
        dtype=np.float32,
    )
    angles = np.arctan2(corners[:, 1], corners[:, 0])
    return float(np.min(angles)), float(np.max(angles))


def _future_xy(state: np.ndarray, horizon_s: float = 2.0) -> np.ndarray:
    x = float(state[AgentIndex.X])
    y = float(state[AgentIndex.Y])
    heading = float(state[AgentIndex.HEADING])
    speed = float(max(state[AgentIndex.VELOCITY], 0.0))
    return np.array(
        [x + math.cos(heading) * speed * horizon_s, y + math.sin(heading) * speed * horizon_s],
        dtype=np.float32,
    )


def strict_forward_ghost_probe_check(
    scene: SledgeVector,
    forward_x_min: float,
    forward_x_max: float,
    corridor_half_width: float,
    min_forward_corridor: float,
    min_blind_spot: float,
    pedestrian_future_horizon_s: float = 2.0,
) -> Dict[str, object]:
    vehicles = _valid_states(scene.vehicles)
    static_objects = _valid_states(scene.static_objects)
    pedestrians = _valid_states(scene.pedestrians)

    occ_candidates: List[Tuple[str, np.ndarray, float, float]] = []
    for state in vehicles:
        x = float(state[AgentIndex.X])
        y = float(state[AgentIndex.Y])
        if forward_x_min <= x <= min(18.0, forward_x_max) and abs(y) >= corridor_half_width + 0.8:
            occ_candidates.append(("vehicle", state, float(state[AgentIndex.LENGTH]), float(state[AgentIndex.WIDTH])))

    for state in static_objects:
        x = float(state[StaticObjectIndex.X])
        y = float(state[StaticObjectIndex.Y])
        if forward_x_min <= x <= min(18.0, forward_x_max) and abs(y) >= corridor_half_width + 0.8:
            fake = np.zeros(6, dtype=np.float32)
            fake[AgentIndex.X] = state[StaticObjectIndex.X]
            fake[AgentIndex.Y] = state[StaticObjectIndex.Y]
            fake[AgentIndex.HEADING] = state[StaticObjectIndex.HEADING]
            fake[AgentIndex.LENGTH] = state[StaticObjectIndex.LENGTH]
            fake[AgentIndex.WIDTH] = state[StaticObjectIndex.WIDTH]
            fake[AgentIndex.VELOCITY] = 0.0
            occ_candidates.append(("static", fake, float(fake[AgentIndex.LENGTH]), float(fake[AgentIndex.WIDTH])))

    if len(occ_candidates) == 0 or len(pedestrians) == 0:
        return {
            "passed": False,
            "front_occluder_found": len(occ_candidates) > 0,
            "pedestrian_found": len(pedestrians) > 0,
            "forward_corridor_score": 0.0,
            "blind_spot_score": 0.0,
            "reason": "missing occluder or pedestrian",
        }

    best_forward_corridor = 0.0
    best_blind_spot = 0.0
    best_pair = None

    for _, occ_state, occ_len, occ_wid in occ_candidates:
        occ_xy = occ_state[AgentIndex.POINT]
        occ_a0, occ_a1 = _angular_interval_from_bbox(occ_xy, occ_len, occ_wid)
        occ_side = np.sign(float(occ_state[AgentIndex.Y])) if abs(float(occ_state[AgentIndex.Y])) > 1e-3 else 0.0

        for ped_state in pedestrians:
            ped_xy = ped_state[AgentIndex.POINT]
            ped_side = np.sign(float(ped_state[AgentIndex.Y])) if abs(float(ped_state[AgentIndex.Y])) > 1e-3 else 0.0
            if occ_side != 0.0 and ped_side != occ_side:
                continue
            if not (float(occ_xy[0]) - 1.5 <= float(ped_xy[0]) <= float(occ_xy[0]) + 5.0):
                continue

            future_xy = _future_xy(ped_state, horizon_s=pedestrian_future_horizon_s)
            forward_score = 1.0 if (
                forward_x_min <= float(future_xy[0]) <= forward_x_max
                and abs(float(future_xy[1])) <= corridor_half_width
            ) else 0.0

            ped_a0, ped_a1 = _angular_interval_from_bbox(ped_xy, 0.8, 0.8)
            overlap = _interval_overlap_ratio(occ_a0, occ_a1, ped_a0, ped_a1)
            blind_score = float(overlap) if forward_score > 0.0 else 0.0

            if (forward_score, blind_score) > (best_forward_corridor, best_blind_spot):
                best_forward_corridor = forward_score
                best_blind_spot = blind_score
                best_pair = {
                    "occluder_xy": occ_xy.tolist(),
                    "pedestrian_xy": ped_xy.tolist(),
                    "pedestrian_future_xy": future_xy.tolist(),
                }

    passed = best_forward_corridor >= min_forward_corridor and best_blind_spot >= min_blind_spot
    return {
        "passed": bool(passed),
        "front_occluder_found": True,
        "pedestrian_found": True,
        "forward_corridor_score": float(best_forward_corridor),
        "blind_spot_score": float(best_blind_spot),
        "best_pair": best_pair,
        "reason": "ok" if passed else "forward corridor / blind spot constraint not met after denoising",
    }


def _roi_union(rois: List[SceneEditROI]) -> Optional[SceneEditROI]:
    if not rois:
        return None
    return SceneEditROI(
        x_min=min(r.x_min for r in rois),
        y_min=min(r.y_min for r in rois),
        x_max=max(r.x_max for r in rois),
        y_max=max(r.y_max for r in rois),
        tag="union",
    )


def _find_tagged_roi(rois: List[SceneEditROI], tag: str) -> Optional[SceneEditROI]:
    for roi in rois:
        if getattr(roi, "tag", "") == tag:
            return roi
    return None


def _corridor_roi_from_ped_to_future(
    ped_roi: SceneEditROI,
    future_xy: np.ndarray,
    band_half_width: float,
) -> SceneEditROI:
    ped_center_x = 0.5 * (ped_roi.x_min + ped_roi.x_max)
    ped_center_y = 0.5 * (ped_roi.y_min + ped_roi.y_max)
    x_min = min(ped_center_x, float(future_xy[0])) - 0.6
    x_max = max(ped_center_x, float(future_xy[0])) + 0.6
    y_mid = float(future_xy[1])
    y_min = min(ped_center_y, y_mid) - band_half_width
    y_max = max(ped_center_y, y_mid) + band_half_width
    return SceneEditROI(x_min=x_min, y_min=y_min, x_max=x_max, y_max=y_max, tag="pedestrian_corridor")


def _bridge_roi_between(roi_a: SceneEditROI, roi_b: SceneEditROI, extra_width: float) -> SceneEditROI:
    ax = 0.5 * (roi_a.x_min + roi_a.x_max)
    ay = 0.5 * (roi_a.y_min + roi_a.y_max)
    bx = 0.5 * (roi_b.x_min + roi_b.x_max)
    by = 0.5 * (roi_b.y_min + roi_b.y_max)
    return SceneEditROI(
        x_min=min(ax, bx) - extra_width,
        y_min=min(ay, by) - extra_width,
        x_max=max(ax, bx) + extra_width,
        y_max=max(ay, by) + extra_width,
        tag="occluder_pedestrian_bridge",
    )


def extend_preserved_rois_for_forward_constraint(
    base_rois: List[SceneEditROI],
    edited_vector: SledgeVector,
    strict_forward_result: Dict[str, object],
    corridor_half_width: float,
    forward_x_min: float,
    forward_x_max: float,
    future_horizon_s: float,
    corridor_band_half_width: float,
    bridge_extra_width: float,
) -> List[SceneEditROI]:
    rois = list(base_rois)
    ped_roi = _find_tagged_roi(rois, "pedestrian")
    occ_roi = _find_tagged_roi(rois, "occluder")

    # If ROI tags are missing, fall back to unioned entities.
    if ped_roi is None:
        ped_states = _valid_states(edited_vector.pedestrians)
        if len(ped_states) > 0:
            p = ped_states[0]
            w = 0.9
            ped_roi = SceneEditROI(
                x_min=float(p[AgentIndex.X] - w),
                y_min=float(p[AgentIndex.Y] - w),
                x_max=float(p[AgentIndex.X] + w),
                y_max=float(p[AgentIndex.Y] + w),
                tag="pedestrian",
            )
            rois.append(ped_roi)

    if occ_roi is None:
        # Try to recover from strict_forward best pair or from vehicle/static closest to that point.
        best_pair = strict_forward_result.get("best_pair", None)
        if isinstance(best_pair, dict) and "occluder_xy" in best_pair:
            occ_xy = np.asarray(best_pair["occluder_xy"], dtype=np.float32)
            occ_roi = SceneEditROI(
                x_min=float(occ_xy[0] - 3.5),
                y_min=float(occ_xy[1] - 2.5),
                x_max=float(occ_xy[0] + 3.5),
                y_max=float(occ_xy[1] + 2.5),
                tag="occluder",
            )
            rois.append(occ_roi)

    if ped_roi is None:
        return rois

    ped_states = _valid_states(edited_vector.pedestrians)
    if len(ped_states) > 0:
        ped_state = ped_states[0]
        future_xy = _future_xy(ped_state, horizon_s=future_horizon_s)

        # Clamp future anchor into the forward corridor so that the mask preserves the crucial entry area.
        future_xy = np.array(
            [
                np.clip(float(future_xy[0]), forward_x_min, forward_x_max),
                np.clip(float(future_xy[1]), -corridor_half_width, corridor_half_width),
            ],
            dtype=np.float32,
        )
        rois.append(_corridor_roi_from_ped_to_future(ped_roi, future_xy, corridor_band_half_width))

        # Also preserve a small anchor in the central corridor around the future crossing point.
        rois.append(
            SceneEditROI(
                x_min=float(future_xy[0] - 1.2),
                y_min=float(future_xy[1] - corridor_band_half_width),
                x_max=float(future_xy[0] + 1.2),
                y_max=float(future_xy[1] + corridor_band_half_width),
                tag="forward_entry_anchor",
            )
        )

    if occ_roi is not None:
        rois.append(_bridge_roi_between(occ_roi, ped_roi, bridge_extra_width))

    # Add union ROI as a final safety blanket for local structure.
    union_roi = _roi_union(rois)
    if union_roi is not None:
        union_roi.tag = "strict_forward_union"
        rois.append(union_roi)

    return rois


def build_preserve_mask(
    rois: List[SceneEditROI],
    config: SledgeConfig,
    latent_shape: torch.Size,
    device: str,
    dilation: int,
) -> torch.Tensor:
    _, _, latent_h, latent_w = latent_shape
    pixel_width, pixel_height = config.pixel_frame
    raster_mask = np.zeros((pixel_width, pixel_height), dtype=np.float32)

    for roi in rois:
        x_min = int(np.floor((roi.x_min + config.frame[0] / 2.0) / config.pixel_size))
        x_max = int(np.ceil((roi.x_max + config.frame[0] / 2.0) / config.pixel_size))
        y_min = int(np.floor((roi.y_min + config.frame[1] / 2.0) / config.pixel_size))
        y_max = int(np.ceil((roi.y_max + config.frame[1] / 2.0) / config.pixel_size))

        x_min, x_max = max(0, x_min), min(pixel_width, x_max)
        y_min, y_max = max(0, y_min), min(pixel_height, y_max)
        raster_mask[x_min:x_max, y_min:y_max] = 1.0

    mask = torch.from_numpy(raster_mask).view(1, 1, pixel_width, pixel_height).to(device)
    mask = F.interpolate(mask, size=(latent_h, latent_w), mode="nearest")
    if dilation > 0:
        kernel = 1 + 2 * dilation
        mask = F.max_pool2d(mask, kernel_size=kernel, stride=1, padding=dilation)
    return mask.clamp(0.0, 1.0)


class SemanticBatchRunner:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.out_root = Path(args.output)
        self.out_root.mkdir(parents=True, exist_ok=True)

        self.cfg = OmegaConf.load(args.config)
        self.cfg.autoencoder_checkpoint = args.autoencoder_checkpoint
        self.cfg.diffusion_checkpoint = args.diffusion_checkpoint

        ae_cfg_dict = OmegaConf.to_container(self.cfg.autoencoder_model.config, resolve=True)
        if not isinstance(ae_cfg_dict, dict):
            raise TypeError(f"Expected autoencoder_model.config to resolve to dict, got {type(ae_cfg_dict)}")
        filtered = {k: v for k, v in ae_cfg_dict.items() if k in RVAEConfig.__annotations__}
        self.ae_config = RVAEConfig(**filtered)

        self.prompt_parser = NaturalLanguagePromptParser()
        self.scene_editor = SemanticSceneEditor()
        self.alignment_evaluator = PromptAlignmentEvaluator()

        self.autoencoder_model = build_autoencoder_torch_module_wrapper(self.cfg)
        if hasattr(self.autoencoder_model, "eval"):
            self.autoencoder_model.eval()

        self.pipeline = build_pipeline_from_checkpoint(self.cfg)
        self.pipeline.to(args.device)
        if hasattr(self.pipeline, "transformer") and self.pipeline.transformer is not None:
            self.pipeline.transformer.eval()
        if hasattr(self.pipeline, "unet") and self.pipeline.unet is not None:
            self.pipeline.unet.eval()

        self.scenario_cache_root = self._resolve_scenario_cache_root(args.scenario_cache_root)
        self.scenario_cache_root.mkdir(parents=True, exist_ok=True)

        # parse low-noise schedule once
        raw_seq = [s.strip() for s in str(args.low_noise_start_step_seq).split(",") if s.strip()]
        if raw_seq:
            self.start_step_candidates = [max(1, int(v)) for v in raw_seq]
        else:
            self.start_step_candidates = [max(1, int(args.start_timestep_index))]
        if args.start_timestep_index not in self.start_step_candidates:
            self.start_step_candidates.append(max(1, int(args.start_timestep_index)))
        self.start_step_candidates = sorted(list(set(self.start_step_candidates)))

    def _resolve_scenario_cache_root(self, override: Optional[str]) -> Path:
        if override:
            return Path(override)
        sledge_exp_root = os.environ.get("SLEDGE_EXP_ROOT")
        if not sledge_exp_root:
            raise EnvironmentError(
                "SLEDGE_EXP_ROOT is not set. Export it or pass --scenario-cache-root explicitly."
            )
        return Path(sledge_exp_root) / "caches" / "scenario_cache"

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
        rel = Path(*parts[-3:]) if len(parts) >= 3 else Path(scene_path.parent.name)
        if self.args.output_layout == "flat":
            rel = Path(f"{index:06d}_{scene_path.parent.name}")
        return self.scenario_cache_root / rel

    def _attempt_generation(
        self,
        init_latents: torch.Tensor,
        map_id: int,
        preserve_mask: torch.Tensor,
        attempt_idx: int,
    ):
        # Conservative schedule: try the smallest corruption first.
        start_idx = self.start_step_candidates[min(attempt_idx, len(self.start_step_candidates) - 1)]
        dilation = self.args.mask_dilation + min(attempt_idx, 2)
        mask_used = preserve_mask
        if dilation != self.args.mask_dilation:
            kernel = 1 + 2 * dilation
            mask_used = F.max_pool2d(preserve_mask, kernel_size=kernel, stride=1, padding=dilation).clamp(0.0, 1.0)

        with torch.no_grad():
            denoised_vectors, final_latents = self.pipeline(
                class_labels=[map_id],
                num_inference_timesteps=self.args.num_inference_timesteps,
                guidance_scale=self.args.guidance_scale,
                num_classes=self.cfg.num_classes,
                init_latents=init_latents,
                start_timestep_index=start_idx,
                preserve_mask=mask_used,
                return_latents=True,
            )

        return denoised_vectors[0].torch_to_numpy(apply_sigmoid=True), final_latents, start_idx, dilation, mask_used

    def run_one(self, scene_path: Path, out_dir: Path, index: int = 1) -> Dict[str, object]:
        out_dir.mkdir(parents=True, exist_ok=True)

        raw_scene, source_format = load_raw_scene(scene_path)
        prompt_spec = self.prompt_parser.parse(self.args.prompt)
        map_id = resolve_map_id(scene_path, self.args.map_id, prompt_spec.map_id)

        edited_raw, edit_result = self.scene_editor.edit(raw_scene, prompt_spec)
        edited_vector, processed_raster = sledge_raw_feature_processing(edited_raw, self.ae_config)

        # Prove edit-stage geometry first and use it to strengthen preserve regions.
        edited_alignment = self.alignment_evaluator.evaluate(edited_vector, prompt_spec)
        strict_forward_before = strict_forward_ghost_probe_check(
            edited_vector,
            forward_x_min=self.args.strict_forward_x_min,
            forward_x_max=self.args.strict_forward_x_max,
            corridor_half_width=self.args.strict_corridor_half_width,
            min_forward_corridor=self.args.strict_min_forward_corridor,
            min_blind_spot=self.args.strict_min_blind_spot,
            pedestrian_future_horizon_s=self.args.pedestrian_future_horizon_s,
        )

        strengthened_rois = extend_preserved_rois_for_forward_constraint(
            base_rois=edit_result.preserved_rois,
            edited_vector=edited_vector,
            strict_forward_result=strict_forward_before,
            corridor_half_width=self.args.strict_corridor_half_width,
            forward_x_min=self.args.strict_forward_x_min,
            forward_x_max=self.args.strict_forward_x_max,
            future_horizon_s=self.args.pedestrian_future_horizon_s,
            corridor_band_half_width=self.args.preserve_corridor_band_half_width,
            bridge_extra_width=self.args.preserve_occ_ped_bridge_extra_width,
        )

        init_latents = encode_raster(self.autoencoder_model, processed_raster, self.args.device)
        preserve_mask = build_preserve_mask(
            rois=strengthened_rois,
            config=self.ae_config,
            latent_shape=init_latents.shape,
            device=self.args.device,
            dilation=self.args.mask_dilation,
        )

        best = None
        best_score = -1e9
        max_attempts = max(1, int(self.args.resample_attempts))

        for attempt_idx in range(max_attempts):
            denoised_vector, final_latents, start_idx, dilation, mask_used = self._attempt_generation(
                init_latents, map_id, preserve_mask, attempt_idx
            )
            alignment = self.alignment_evaluator.evaluate(denoised_vector, prompt_spec)
            strict_forward = strict_forward_ghost_probe_check(
                denoised_vector,
                forward_x_min=self.args.strict_forward_x_min,
                forward_x_max=self.args.strict_forward_x_max,
                corridor_half_width=self.args.strict_corridor_half_width,
                min_forward_corridor=self.args.strict_min_forward_corridor,
                min_blind_spot=self.args.strict_min_blind_spot,
                pedestrian_future_horizon_s=self.args.pedestrian_future_horizon_s,
            )
            alignment_ok = alignment.total >= self.args.alignment_threshold
            strict_ok = True if prompt_spec.scenario_type != "ghost_probe" else bool(strict_forward["passed"])

            candidate = {
                "denoised_vector": denoised_vector,
                "final_latents": final_latents,
                "alignment": alignment,
                "strict_forward": strict_forward,
                "alignment_ok": bool(alignment_ok),
                "strict_ok": bool(strict_ok),
                "attempt_idx": attempt_idx,
                "start_timestep_index": start_idx,
                "mask_dilation": dilation,
                "mask_sum": float(mask_used.sum().item()),
            }

            # Prioritize strict-forward passing, then alignment, then blind-spot strength.
            rank_score = (
                3.0 * float(strict_ok)
                + 2.0 * float(strict_forward.get("forward_corridor_score", 0.0))
                + 1.5 * float(strict_forward.get("blind_spot_score", 0.0))
                + float(alignment.total)
            )
            if rank_score > best_score:
                best = candidate
                best_score = rank_score
            if alignment_ok and strict_ok:
                best = candidate
                break

        assert best is not None
        denoised_vector: SledgeVector = best["denoised_vector"]
        final_latents = best["final_latents"]
        alignment = best["alignment"]
        strict_forward_after = best["strict_forward"]
        accepted = bool(best["alignment_ok"] and best["strict_ok"])

        save_raw_scene(out_dir / "edited_sledge_raw", edited_raw, source_format=source_format)
        save_json(out_dir / "prompt_spec.json", prompt_spec.to_dict())
        edit_report = edit_result.to_dict()
        edit_report["preserved_rois"] = [roi.__dict__ for roi in strengthened_rois]
        save_json(out_dir / "edit_report.json", edit_report)
        save_json(
            out_dir / "prompt_alignment.json",
            {
                **alignment.to_dict(),
                "accepted": bool(accepted),
                "map_id": int(map_id),
                "map_name": MAP_ID_TO_NAME.get(map_id, "unknown"),
                "source_format": source_format,
                "strict_forward": strict_forward_after,
                "strict_forward_before_diffusion": strict_forward_before,
                "edited_alignment_before_diffusion": edited_alignment.to_dict(),
                "attempt_idx": int(best["attempt_idx"]),
                "used_start_timestep_index": int(best["start_timestep_index"]),
                "used_mask_dilation": int(best["mask_dilation"]),
                "preserve_mask_sum": float(best["mask_sum"]),
            },
        )

        visual_raster = processed_raster.to_feature_tensor()
        visual_raster = SledgeRaster(visual_raster.data.unsqueeze(0).cpu())
        save_image(out_dir / "edited_raster.png", get_sledge_raster(visual_raster, self.ae_config.pixel_frame))
        save_image(out_dir / "edited_vector.png", get_sledge_vector_as_raster(edited_vector, self.ae_config))
        save_image(out_dir / "denoised_vector.png", get_sledge_vector_as_raster(denoised_vector, self.ae_config))

        scenario_vector_path = None
        if accepted or self.args.allow_write_failed_scenario_cache:
            scenario_cache_dir = self._scenario_cache_dir(scene_path, index)
            scenario_cache_dir.mkdir(parents=True, exist_ok=True)
            scenario_vector_payload = feature_to_raw_scene_dict(denoised_vector)
            scenario_vector_path = save_gz_pickle(scenario_cache_dir / "sledge_vector", scenario_vector_payload)

        if self.args.save_latents:
            torch.save(init_latents.detach().cpu(), out_dir / "init_latents.pt")
            torch.save(final_latents.detach().cpu(), out_dir / "final_latents.pt")
            torch.save(preserve_mask.detach().cpu(), out_dir / "preserve_mask.pt")

        summary = {
            "scene_path": str(scene_path),
            "output_dir": str(out_dir),
            "scenario_cache_vector_path": str(scenario_vector_path) if scenario_vector_path else None,
            "map_id": int(map_id),
            "source_format": source_format,
            "prompt_type": prompt_spec.scenario_type,
            "edited_alignment_before_diffusion": float(edited_alignment.total),
            "alignment_total": float(alignment.total),
            "accepted": bool(accepted),
            "strict_forward_before_diffusion": strict_forward_before,
            "strict_forward": strict_forward_after,
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
        summary_rows: List[Dict[str, object]] = []

        for index, scene_path in enumerate(scene_paths, start=1):
            out_dir = self._scene_output_dir(scene_path, index)
            scenario_cache_dir = self._scenario_cache_dir(scene_path, index)
            marker = out_dir / "prompt_alignment.json"
            scenario_marker = scenario_cache_dir / "sledge_vector.gz"
            if self.args.skip_existing and marker.exists() and scenario_marker.exists():
                print(f"[{index}/{total}] skipped: {scene_path}")
                continue

            print(f"[{index}/{total}] processing: {scene_path}")
            try:
                row = self.run_one(scene_path, out_dir, index=index)
                summary_rows.append(row)
                print(
                    f"[{index}/{total}] done: {scene_path} | "
                    f"edited_alignment={row['edited_alignment_before_diffusion']:.4f} | "
                    f"alignment={row['alignment_total']:.4f} | "
                    f"accepted={row['accepted']} | "
                    f"strict_forward_before={row['strict_forward_before_diffusion'].get('passed', False)} | "
                    f"strict_forward_after={row['strict_forward'].get('passed', False)}"
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
    runner = SemanticBatchRunner(args)
    if args.input:
        runner.run_one(Path(args.input), Path(args.output), index=1)
        return
    runner.run_batch()


if __name__ == "__main__":
    main()
