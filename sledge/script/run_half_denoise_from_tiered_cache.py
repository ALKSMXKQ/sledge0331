from __future__ import annotations

import argparse
import json
import math
import os
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf

from sledge.autoencoder.modeling.models.rvae.rvae_config import RVAEConfig
from sledge.autoencoder.preprocessing.feature_builders.sledge.sledge_feature_processing import (
    sledge_raw_feature_processing,
)
from sledge.autoencoder.preprocessing.feature_builders.sledge.sledge_utils import (
    coords_in_frame,
    coords_to_pixel,
    pixel_in_frame,
)
from sledge.autoencoder.preprocessing.features.map_id_feature import MAP_ID_TO_NAME
from sledge.autoencoder.preprocessing.features.sledge_raster_feature import SledgeRaster
from sledge.autoencoder.preprocessing.features.sledge_vector_feature import (
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
from sledge.semantic_control import NaturalLanguagePromptParser, PromptAlignmentEvaluator
from sledge.semantic_control.io import (
    feature_to_raw_scene_dict,
    load_raw_scene,
    save_gz_pickle,
    save_json,
)

DEFAULT_ALIGNMENT_THRESHOLD = 0.70
LABEL_THRESH = 0.3

# Explicit indices to avoid enum/property surprises when indexing numpy arrays.
AGENT_X = 0
AGENT_Y = 1
AGENT_HEADING = 2
AGENT_WIDTH = 3
AGENT_LENGTH = 4
AGENT_VELOCITY = 5
AGENT_POINT = slice(0, 2)

LINE_CH = slice(0, 2)
VEHICLE_CH = slice(2, 4)
PEDESTRIAN_CH = slice(4, 6)
STATIC_CH = slice(6, 8)
GREEN_CH = slice(8, 10)
RED_CH = slice(10, 12)
NUM_RASTER_CH = 12


# -----------------------------
# CLI
# -----------------------------
def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Alternating low-noise diffusion and minimal semantic projection from an already edited tiered crossing cache. "
            "Only diffusion-generated scenes that preserve semantics and pass compliance checks are saved."
        )
    )
    parser.add_argument("--original-dir", required=True, help="Root of original raw cache")
    parser.add_argument("--edited-dir", required=True, help="Root of already edited tiered_crossing_raw_cache")
    parser.add_argument("--output", required=True, help="Output directory for reports / debug artifacts")
    parser.add_argument("--config", required=True, help="Expanded OmegaConf yaml")
    parser.add_argument("--autoencoder-checkpoint", required=True, help="RVAE checkpoint path")
    parser.add_argument("--diffusion-checkpoint", required=True, help="DiT / diffusion pipeline checkpoint path")
    parser.add_argument(
        "--scenario-cache-root",
        default=None,
        help="Where to write final successful diffusion sledge_vector.gz. Defaults to $SLEDGE_EXP_ROOT/caches/scenario_cache_half_denoise_best",
    )
    parser.add_argument("--map-id", type=int, default=None)
    parser.add_argument("--glob-pattern", default="**/sledge_raw.gz")
    parser.add_argument("--max-scenes", type=int, default=None)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--output-layout", choices=["mirror", "flat"], default="mirror")

    parser.add_argument("--num-inference-timesteps", type=int, default=24)
    parser.add_argument("--guidance-scale", type=float, default=4.0)
    parser.add_argument("--round-start-step-seq", default="14,10,6")
    parser.add_argument("--low-noise-start-step-seq", default=None, help="Backward-compatible alias; ignored when --round-start-step-seq is provided")
    parser.add_argument("--repair-attempts", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)

    parser.add_argument("--alignment-threshold", type=float, default=DEFAULT_ALIGNMENT_THRESHOLD)
    parser.add_argument("--min-preservation-ratio", type=float, default=0.95)

    parser.add_argument("--diff-threshold", type=float, default=1e-4)
    parser.add_argument("--diff-mask-dilation", type=int, default=3)
    parser.add_argument("--roi-mask-dilation", type=int, default=2)

    parser.add_argument("--pedestrian-roi-strength", type=float, default=1.00)
    parser.add_argument("--roadside-anchor-strength", type=float, default=1.00)
    parser.add_argument("--lane-anchor-strength", type=float, default=1.00)
    parser.add_argument("--crossing-corridor-strength", type=float, default=0.95)
    parser.add_argument("--generic-roi-strength", type=float, default=0.90)

    parser.add_argument("--projection-inner-iters", type=int, default=2)
    parser.add_argument("--projection-x-alpha", type=float, default=0.35)
    parser.add_argument("--projection-y-alpha", type=float, default=0.45)
    parser.add_argument("--projection-heading-alpha", type=float, default=0.40)
    parser.add_argument("--projection-velocity-alpha", type=float, default=0.40)
    parser.add_argument("--projection-size-alpha", type=float, default=0.20)
    parser.add_argument("--projection-max-pos-shift-m", type=float, default=1.8)
    parser.add_argument("--projection-max-heading-shift-rad", type=float, default=0.7)
    parser.add_argument("--projection-max-speed-delta", type=float, default=0.8)
    parser.add_argument("--projection-match-max-dist", type=float, default=10.0)

    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--save-latents", action="store_true")
    parser.add_argument("--save-visuals", action="store_true")
    return parser


# -----------------------------
# Generic helpers
# -----------------------------
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


def load_json(path: Path) -> Dict[str, object]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def encode_raster(autoencoder_model, raster: SledgeRaster, device: str) -> torch.Tensor:
    raster_tensor = raster.to_feature_tensor().data.unsqueeze(0).to(device)
    encoder = autoencoder_model.get_encoder().to(device)
    encoder.eval()
    with torch.no_grad():
        latent_dist = encoder(raster_tensor)
    return latent_dist.mu


def build_raster_diff_mask(
    original_raster: SledgeRaster,
    edited_raster: SledgeRaster,
    latent_shape: torch.Size,
    device: str,
    diff_threshold: float,
    dilation: int,
) -> torch.Tensor:
    original = original_raster.to_feature_tensor().data.float().unsqueeze(0).to(device)
    edited = edited_raster.to_feature_tensor().data.float().unsqueeze(0).to(device)
    diff = (edited - original).abs().sum(dim=1, keepdim=True)
    mask = (diff > diff_threshold).float()
    if dilation > 0:
        kernel = 1 + 2 * dilation
        mask = F.max_pool2d(mask, kernel_size=kernel, stride=1, padding=dilation)
    mask = F.interpolate(mask, size=(latent_shape[2], latent_shape[3]), mode="nearest")
    return mask.clamp(0.0, 1.0)


def _roi_strength(tag: str, args: argparse.Namespace) -> float:
    tag = (tag or "").lower()
    if tag == "pedestrian":
        return float(args.pedestrian_roi_strength)
    if tag == "roadside_spawn_anchor":
        return float(args.roadside_anchor_strength)
    if tag == "lane_edge_conflict_anchor":
        return float(args.lane_anchor_strength)
    if tag == "crossing_corridor":
        return float(args.crossing_cordidor_strength) if hasattr(args, "crossing_cordidor_strength") else float(args.crossing_corridor_strength)
    return float(args.generic_roi_strength)


def build_roi_soft_mask(
    roi_dicts: List[Dict[str, float]],
    config: SledgeConfig,
    latent_shape: torch.Size,
    device: str,
    dilation: int,
    args: argparse.Namespace,
) -> torch.Tensor:
    _, _, latent_h, latent_w = latent_shape
    pixel_width, pixel_height = config.pixel_frame
    raster_mask = np.zeros((pixel_width, pixel_height), dtype=np.float32)

    for roi in roi_dicts:
        strength = _roi_strength(str(roi.get("tag", "")), args)
        x_min = int(np.floor((float(roi["x_min"]) + config.frame[0] / 2.0) / config.pixel_size))
        x_max = int(np.ceil((float(roi["x_max"]) + config.frame[0] / 2.0) / config.pixel_size))
        y_min = int(np.floor((float(roi["y_min"]) + config.frame[1] / 2.0) / config.pixel_size))
        y_max = int(np.ceil((float(roi["y_max"]) + config.frame[1] / 2.0) / config.pixel_size))
        x_min, x_max = max(0, x_min), min(pixel_width, x_max)
        y_min, y_max = max(0, y_min), min(pixel_height, y_max)
        if x_min >= x_max or y_min >= y_max:
            continue
        raster_mask[x_min:x_max, y_min:y_max] = np.maximum(raster_mask[x_min:x_max, y_min:y_max], strength)

    mask = torch.from_numpy(raster_mask).view(1, 1, pixel_width, pixel_height).to(device)
    mask = F.interpolate(mask, size=(latent_h, latent_w), mode="nearest")
    if dilation > 0:
        kernel = 1 + 2 * dilation
        mask = F.max_pool2d(mask, kernel_size=kernel, stride=1, padding=dilation)
    return mask.clamp(0.0, 1.0)


def make_simulation_compatible_vector(processed_vector: SledgeVector, edited_raw) -> SledgeVector:
    raw_ego_states = np.asarray(edited_raw.ego.states)
    raw_ego_mask = np.asarray(edited_raw.ego.mask)
    ego_speed = float(raw_ego_states.reshape(-1)[0]) if raw_ego_states.size > 0 else 0.0
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


def basic_scene_compliance(vector: SledgeVector) -> Dict[str, object]:
    issues: List[str] = []

    def _check_elem(name: str, elem) -> None:
        states = np.asarray(elem.states)
        mask = np.asarray(elem.mask)
        if np.isnan(states).any() or np.isinf(states).any():
            issues.append(f"{name}: invalid numeric values")
        if np.isnan(mask).any() or np.isinf(mask).any():
            issues.append(f"{name}: invalid mask values")

    _check_elem("lines", vector.lines)
    _check_elem("vehicles", vector.vehicles)
    _check_elem("pedestrians", vector.pedestrians)
    _check_elem("static_objects", vector.static_objects)
    _check_elem("green_lights", vector.green_lights)
    _check_elem("red_lights", vector.red_lights)
    _check_elem("ego", vector.ego)

    ped_mask = np.asarray(vector.pedestrians.mask).astype(bool)
    if not np.any(ped_mask):
        issues.append("no valid pedestrian after repair")

    ego_states = np.asarray(vector.ego.states).reshape(-1)
    if ego_states.size == 0:
        issues.append("missing ego state")

    return {"compliant": len(issues) == 0, "issues": issues}


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


def summarize_crossing_semantics(alignment: object, prompt_spec: object, threshold: float) -> Dict[str, object]:
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

    semantic_pass = all(checks.values()) if crossing_prompt else total >= float(threshold)
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


# -----------------------------
# Rasterization from processed vector
# -----------------------------
def _valid_mask(mask: np.ndarray) -> np.ndarray:
    mask_arr = np.asarray(mask)
    if mask_arr.ndim == 0:
        return np.asarray([bool(mask_arr)])
    return (mask_arr >= LABEL_THRESH).astype(bool)


def _oriented_box_corners_xy(x: float, y: float, heading: float, length: float, width: float) -> np.ndarray:
    half_l = 0.5 * float(length)
    half_w = 0.5 * float(width)
    local = np.asarray(
        [
            [half_l, half_w],
            [half_l, -half_w],
            [-half_l, -half_w],
            [-half_l, half_w],
        ],
        dtype=np.float32,
    )
    c, s = math.cos(float(heading)), math.sin(float(heading))
    rot = np.asarray([[c, -s], [s, c]], dtype=np.float32)
    world = local @ rot.T
    world[:, 0] += float(x)
    world[:, 1] += float(y)
    return world


def _polygon_mask_from_xy(corners_xy: np.ndarray, config: SledgeConfig) -> np.ndarray:
    pixel_width, pixel_height = config.pixel_frame
    corner_idx = coords_to_pixel(corners_xy.astype(np.float32), config.frame, config.pixel_size)
    raster_mask = np.zeros((pixel_width, pixel_height), dtype=np.float32)
    cv2.fillPoly(raster_mask, [corner_idx.astype(np.int32)], color=1.0, lineType=cv2.LINE_AA)
    # Match repository orientation convention used by raster_mask_oriented_box().
    raster_mask = np.rot90(raster_mask)[::-1]
    return raster_mask > 0


def _rasterize_processed_lines(lines_elem: SledgeVectorElement, config: SledgeConfig) -> np.ndarray:
    pixel_height, pixel_width = config.pixel_frame
    raster = np.zeros((pixel_height, pixel_width, 2), dtype=np.float32)
    states = np.asarray(lines_elem.states, dtype=np.float32)
    masks = _valid_mask(np.asarray(lines_elem.mask))
    if states.ndim == 2:
        states = states[None, ...]
    for line, valid in zip(states, masks):
        if not valid:
            continue
        pts = np.asarray(line, dtype=np.float32)
        frame_mask = coords_in_frame(pts[..., :2], config.frame)
        pts = pts[frame_mask]
        if len(pts) < 2:
            continue
        diffs = np.diff(pts, axis=0, prepend=pts[:1])
        norms = np.linalg.norm(diffs, axis=-1, keepdims=True)
        norms = np.maximum(norms, 1e-6)
        dirs = diffs / norms
        values = 0.5 * (dirs + 1.0)
        pixel_coords = coords_to_pixel(pts[..., :2], config.frame, config.pixel_size)
        pixel_mask = pixel_in_frame(pixel_coords, config.pixel_frame)
        pixel_coords, values = pixel_coords[pixel_mask], values[pixel_mask]
        if len(pixel_coords) == 0:
            continue
        raster[pixel_coords[..., 0], pixel_coords[..., 1]] = values
    return raster


def _rasterize_processed_agents(agents_elem: SledgeVectorElement, config: SledgeConfig, max_velocity: float) -> np.ndarray:
    pixel_height, pixel_width = config.pixel_frame
    raster = np.zeros((pixel_height, pixel_width, 2), dtype=np.float32)
    states = np.asarray(agents_elem.states, dtype=np.float32)
    masks = _valid_mask(np.asarray(agents_elem.mask))
    if states.ndim == 1:
        states = states[None, :]
    for state, valid in zip(states, masks):
        if not valid:
            continue
        x = float(state[AGENT_X])
        y = float(state[AGENT_Y])
        heading = float(state[AGENT_HEADING])
        width = float(max(0.1, state[AGENT_WIDTH]))
        length = float(max(0.1, state[AGENT_LENGTH]))
        velocity = float(np.clip(state[AGENT_VELOCITY], 0.0, max_velocity))
        corners = _oriented_box_corners_xy(x, y, heading, length, width)
        raster_mask = _polygon_mask_from_xy(corners, config)
        dx = velocity * math.cos(heading)
        dy = velocity * math.sin(heading)
        raster[raster_mask, 0] = 0.5 * (dx / max_velocity + 1.0)
        raster[raster_mask, 1] = 0.5 * (dy / max_velocity + 1.0)
    return raster


def _rasterize_processed_static(static_elem: SledgeVectorElement, config: SledgeConfig) -> np.ndarray:
    pixel_height, pixel_width = config.pixel_frame
    raster = np.zeros((pixel_height, pixel_width, 2), dtype=np.float32)
    states = np.asarray(static_elem.states, dtype=np.float32)
    masks = _valid_mask(np.asarray(static_elem.mask))
    if states.ndim == 1:
        states = states[None, :]
    for state, valid in zip(states, masks):
        if not valid:
            continue
        x = float(state[0])
        y = float(state[1])
        heading = float(state[2])
        width = float(max(0.1, state[3]))
        length = float(max(0.1, state[4]))
        corners = _oriented_box_corners_xy(x, y, heading, length, width)
        raster_mask = _polygon_mask_from_xy(corners, config)
        raster[raster_mask, 0] = 0.5 * (math.cos(heading) + 1.0)
        raster[raster_mask, 1] = 0.5 * (math.sin(heading) + 1.0)
    return raster


def _rasterize_ego(ego_elem: SledgeVectorElement, config: SledgeConfig, max_velocity: float) -> np.ndarray:
    pixel_height, pixel_width = config.pixel_frame
    raster = np.zeros((pixel_height, pixel_width, 2), dtype=np.float32)
    ego_states = np.asarray(ego_elem.states, dtype=np.float32).reshape(-1)
    speed = float(abs(ego_states[0])) if ego_states.size > 0 else 0.0
    length = 5.2
    width = 2.1
    corners = _oriented_box_corners_xy(0.0, 0.0, 0.0, length, width)
    raster_mask = _polygon_mask_from_xy(corners, config)
    dx = min(speed, max_velocity)
    raster[raster_mask, 0] = 0.5 * (dx / max_velocity + 1.0)
    raster[raster_mask, 1] = 0.5
    return raster


def processed_vector_to_raster(vector: SledgeVector, config: SledgeConfig) -> SledgeRaster:
    pixel_height, pixel_width = config.pixel_frame
    raster = np.zeros((pixel_height, pixel_width, NUM_RASTER_CH), dtype=np.float32)
    raster[..., LINE_CH] = _rasterize_processed_lines(vector.lines, config)
    vehicle_raster = _rasterize_processed_agents(vector.vehicles, config, config.vehicle_max_velocity)
    ego_raster = _rasterize_ego(vector.ego, config, config.vehicle_max_velocity)
    raster[..., VEHICLE_CH] = np.maximum(vehicle_raster, ego_raster)
    raster[..., PEDESTRIAN_CH] = _rasterize_processed_agents(vector.pedestrians, config, config.pedestrian_max_velocity)
    raster[..., STATIC_CH] = _rasterize_processed_static(vector.static_objects, config)
    raster[..., GREEN_CH] = _rasterize_processed_lines(vector.green_lights, config)
    raster[..., RED_CH] = _rasterize_processed_lines(vector.red_lights, config)
    return SledgeRaster(raster)


# -----------------------------
# Minimal semantic projection
# -----------------------------
def _collect_valid_agent_indices(elem: SledgeVectorElement) -> List[int]:
    masks = _valid_mask(np.asarray(elem.mask))
    return [int(i) for i, valid in enumerate(masks) if bool(valid)]


def _score_candidate_pedestrians(evaluator: PromptAlignmentEvaluator, vector: SledgeVector, ego_speed: float, target: Dict[str, float]) -> List[Tuple[int, Dict[str, Any]]]:
    states = np.asarray(vector.pedestrians.states, dtype=np.float32)
    if states.ndim == 1:
        states = states[None, :]
    out: List[Tuple[int, Dict[str, Any]]] = []
    for idx in _collect_valid_agent_indices(vector.pedestrians):
        metrics = evaluator._score_pedestrian_crossing(states[idx], ego_speed, target)  # type: ignore[attr-defined]
        composite = (
            0.15 * metrics["pedestrian_presence_score"]
            + 0.20 * metrics["roadside_emergence_score"]
            + 0.20 * metrics["crossing_direction_score"]
            + 0.30 * metrics["ego_lane_conflict_score"]
            + 0.15 * metrics["immediacy_score"]
        )
        metrics = dict(metrics)
        metrics["composite"] = float(composite)
        out.append((idx, metrics))
    return out


def _best_anchor_pedestrian(evaluator: PromptAlignmentEvaluator, vector: SledgeVector, prompt_spec: Any) -> Optional[Tuple[int, np.ndarray]]:
    severity = getattr(prompt_spec, "severity_level", "moderate") or "moderate"
    target = {
        "mild": {"ttc_peak": 3.5, "ttc_half_width": 1.0},
        "moderate": {"ttc_peak": 2.5, "ttc_half_width": 0.8},
        "aggressive": {"ttc_peak": 1.6, "ttc_half_width": 0.6},
    }.get(str(severity).lower(), {"ttc_peak": 2.5, "ttc_half_width": 0.8})
    ego_speed = evaluator._extract_ego_speed(vector)  # type: ignore[attr-defined]
    scores = _score_candidate_pedestrians(evaluator, vector, ego_speed, target)
    if not scores:
        return None
    best_idx, _ = max(scores, key=lambda item: item[1]["composite"])
    states = np.asarray(vector.pedestrians.states, dtype=np.float32)
    if states.ndim == 1:
        states = states[None, :]
    return best_idx, states[best_idx].copy()


def _match_candidate_ped(anchor_state: np.ndarray, candidate_vector: SledgeVector, max_dist: float) -> Optional[int]:
    states = np.asarray(candidate_vector.pedestrians.states, dtype=np.float32)
    if states.size == 0:
        return None
    if states.ndim == 1:
        states = states[None, :]
    valid_indices = _collect_valid_agent_indices(candidate_vector.pedestrians)
    if not valid_indices:
        return None
    anchor_xy = anchor_state[AGENT_POINT]
    best = None
    best_dist = float("inf")
    for idx in valid_indices:
        dist = float(np.linalg.norm(states[idx, AGENT_POINT] - anchor_xy))
        if dist < best_dist:
            best_dist = dist
            best = idx
    if best is None or best_dist > max_dist:
        return valid_indices[0]
    return best


def _angle_wrap(x: float) -> float:
    while x > math.pi:
        x -= 2.0 * math.pi
    while x < -math.pi:
        x += 2.0 * math.pi
    return x


def _apply_projection_step(candidate_state: np.ndarray, anchor_state: np.ndarray, metrics: Dict[str, float], args: argparse.Namespace) -> np.ndarray:
    out = candidate_state.copy()

    # Base pull toward anchor
    dx = float(anchor_state[AGENT_X] - out[AGENT_X])
    dy = float(anchor_state[AGENT_Y] - out[AGENT_Y])
    d_heading = _angle_wrap(float(anchor_state[AGENT_HEADING] - out[AGENT_HEADING]))
    d_speed = float(anchor_state[AGENT_VELOCITY] - out[AGENT_VELOCITY])

    pos_scale_x = float(args.projection_x_alpha)
    pos_scale_y = float(args.projection_y_alpha)
    heading_scale = float(args.projection_heading_alpha)
    speed_scale = float(args.projection_velocity_alpha)
    size_scale = float(args.projection_size_alpha)

    if float(metrics.get("roadside_emergence_score", 0.0)) < 0.45:
        pos_scale_y *= 1.35
        pos_scale_x *= 1.15
    if float(metrics.get("crossing_direction_score", 0.0)) < 0.45:
        heading_scale *= 1.35
        speed_scale *= 1.15
    if float(metrics.get("ego_lane_conflict_score", 0.0)) < 0.45:
        pos_scale_x *= 1.35
        speed_scale *= 1.25
    if float(metrics.get("immediacy_score", 0.0)) < 0.35:
        speed_scale *= 1.35
        pos_scale_x *= 1.10

    max_pos = float(args.projection_max_pos_shift_m)
    max_heading = float(args.projection_max_heading_shift_rad)
    max_speed = float(args.projection_max_speed_delta)

    out[AGENT_X] += float(np.clip(pos_scale_x * dx, -max_pos, max_pos))
    out[AGENT_Y] += float(np.clip(pos_scale_y * dy, -max_pos, max_pos))
    out[AGENT_HEADING] = _angle_wrap(float(out[AGENT_HEADING] + np.clip(heading_scale * d_heading, -max_heading, max_heading)))
    out[AGENT_VELOCITY] = float(max(0.0, out[AGENT_VELOCITY] + np.clip(speed_scale * d_speed, -max_speed, max_speed)))
    out[AGENT_WIDTH] = float(max(0.1, out[AGENT_WIDTH] + size_scale * (anchor_state[AGENT_WIDTH] - out[AGENT_WIDTH])))
    out[AGENT_LENGTH] = float(max(0.1, out[AGENT_LENGTH] + size_scale * (anchor_state[AGENT_LENGTH] - out[AGENT_LENGTH])))
    return out


def apply_minimal_semantic_projection(
    candidate_vector: SledgeVector,
    anchor_vector: SledgeVector,
    prompt_spec: Any,
    evaluator: PromptAlignmentEvaluator,
    args: argparse.Namespace,
) -> SledgeVector:
    projected = SledgeVector(
        lines=SledgeVectorElement(np.asarray(candidate_vector.lines.states).copy(), np.asarray(candidate_vector.lines.mask).copy()),
        vehicles=SledgeVectorElement(np.asarray(candidate_vector.vehicles.states).copy(), np.asarray(candidate_vector.vehicles.mask).copy()),
        pedestrians=SledgeVectorElement(np.asarray(candidate_vector.pedestrians.states).copy(), np.asarray(candidate_vector.pedestrians.mask).copy()),
        static_objects=SledgeVectorElement(np.asarray(candidate_vector.static_objects.states).copy(), np.asarray(candidate_vector.static_objects.mask).copy()),
        green_lights=SledgeVectorElement(np.asarray(candidate_vector.green_lights.states).copy(), np.asarray(candidate_vector.green_lights.mask).copy()),
        red_lights=SledgeVectorElement(np.asarray(candidate_vector.red_lights.states).copy(), np.asarray(candidate_vector.red_lights.mask).copy()),
        ego=SledgeVectorElement(np.asarray(candidate_vector.ego.states).copy(), np.asarray(candidate_vector.ego.mask).copy()),
    )

    anchor = _best_anchor_pedestrian(evaluator, anchor_vector, prompt_spec)
    if anchor is None:
        return projected
    _, anchor_state = anchor

    ped_states = np.asarray(projected.pedestrians.states, dtype=np.float32)
    ped_masks = np.asarray(projected.pedestrians.mask)
    if ped_states.ndim == 1:
        ped_states = ped_states[None, :]
    if ped_masks.ndim == 0:
        ped_masks = np.asarray([ped_masks])

    match_idx = _match_candidate_ped(anchor_state, projected, float(args.projection_match_max_dist))
    if match_idx is None:
        # No candidate pedestrian survived decoding; inject anchor state into slot 0 softly.
        if len(ped_states) == 0:
            return projected
        match_idx = 0
        ped_masks[match_idx] = 1.0
        ped_states[match_idx] = anchor_state.copy()
    else:
        ped_masks[match_idx] = max(float(ped_masks[match_idx]), 1.0)

    severity = getattr(prompt_spec, "severity_level", "moderate") or "moderate"
    target = {
        "mild": {"ttc_peak": 3.5, "ttc_half_width": 1.0},
        "moderate": {"ttc_peak": 2.5, "ttc_half_width": 0.8},
        "aggressive": {"ttc_peak": 1.6, "ttc_half_width": 0.6},
    }.get(str(severity).lower(), {"ttc_peak": 2.5, "ttc_half_width": 0.8})
    ego_speed = evaluator._extract_ego_speed(projected)  # type: ignore[attr-defined]

    for _ in range(max(1, int(args.projection_inner_iters))):
        metrics = evaluator._score_pedestrian_crossing(ped_states[match_idx], ego_speed, target)  # type: ignore[attr-defined]
        ped_states[match_idx] = _apply_projection_step(ped_states[match_idx], anchor_state, metrics, args)

    projected.pedestrians = SledgeVectorElement(states=ped_states.astype(np.float32), mask=np.asarray(ped_masks))
    return projected


# -----------------------------
# Runner
# -----------------------------
class HalfDenoiseFromTieredCacheRunner:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.original_dir = Path(args.original_dir).resolve()
        self.edited_dir = Path(args.edited_dir).resolve()
        self.out_root = Path(args.output).resolve()
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
        self.alignment_evaluator = PromptAlignmentEvaluator()

        self.autoencoder_model = build_autoencoder_torch_module_wrapper(self.cfg)
        if hasattr(self.autoencoder_model, "eval"):
            self.autoencoder_model.eval()

        self.pipeline = build_pipeline_from_checkpoint(self.cfg)
        self.pipeline.to(args.device)
        if hasattr(self.pipeline, "transformer") and self.pipeline.transformer is not None:
            self.pipeline.transformer.eval()
        self.num_classes = int(self.cfg.get("num_classes", 5))

        seq_str = args.round_start_step_seq if args.round_start_step_seq else args.low_noise_start_step_seq
        raw_seq = [s.strip() for s in str(seq_str).split(",") if s.strip()]
        self.start_step_candidates = [max(1, int(v)) for v in raw_seq] if raw_seq else [14, 10, 6]

        self.scenario_cache_root = self._resolve_scenario_cache_root(args.scenario_cache_root)
        self.scenario_cache_root.mkdir(parents=True, exist_ok=True)

        self.scene_paths = sorted(self.edited_dir.glob(args.glob_pattern))
        if args.max_scenes is not None:
            self.scene_paths = self.scene_paths[: args.max_scenes]

    def _resolve_scenario_cache_root(self, override: Optional[str]) -> Path:
        if override:
            return Path(override)
        sledge_exp_root = os.environ.get("SLEDGE_EXP_ROOT")
        if not sledge_exp_root:
            raise EnvironmentError("SLEDGE_EXP_ROOT is not set. Export it or pass --scenario-cache-root explicitly.")
        return Path(sledge_exp_root) / "caches" / "scenario_cache_half_denoise_best"

    def _scene_output_dir(self, edited_scene_path: Path, index: int) -> Path:
        if self.args.output_layout == "flat":
            stem = edited_scene_path.parent.name
            return self.out_root / f"{index:06d}_{stem}"
        rel = edited_scene_path.parent.relative_to(self.edited_dir)
        return self.out_root / rel

    def _scenario_cache_dir(self, edited_scene_path: Path, index: int) -> Path:
        rel = edited_scene_path.parent.relative_to(self.edited_dir)
        if self.args.output_layout == "flat":
            rel = Path(f"{index:06d}_{edited_scene_path.parent.name}")
        return self.scenario_cache_root / rel

    def _load_prompt_spec(self, edited_scene_dir: Path):
        severity_label_path = edited_scene_dir / "severity_label.json"
        if severity_label_path.exists():
            payload = load_json(severity_label_path)
            prompt = str(payload.get("prompt", "突发的行人横穿马路"))
            return self.prompt_parser.parse(prompt), prompt
        raise FileNotFoundError(f"Missing severity_label.json under {edited_scene_dir}")

    def _attempt_alternating_repair(
        self,
        original_raster: SledgeRaster,
        edited_vector: SledgeVector,
        edited_raster: SledgeRaster,
        edited_raw: Any,
        prompt_spec: Any,
        map_id: int,
        attempt_idx: int,
        scene_index: int,
        roi_dicts: List[Dict[str, float]],
    ) -> Dict[str, Any]:
        current_vector = edited_vector
        current_raster = edited_raster
        current_latents = encode_raster(self.autoencoder_model, current_raster, self.args.device)
        round_rows: List[Dict[str, Any]] = []

        for round_idx, start_idx in enumerate(self.start_step_candidates):
            diff_mask = build_raster_diff_mask(
                original_raster=original_raster,
                edited_raster=current_raster,
                latent_shape=current_latents.shape,
                device=self.args.device,
                diff_threshold=self.args.diff_threshold,
                dilation=self.args.diff_mask_dilation,
            )
            roi_mask = build_roi_soft_mask(
                roi_dicts=roi_dicts,
                config=self.ae_config,
                latent_shape=current_latents.shape,
                device=self.args.device,
                dilation=self.args.roi_mask_dilation,
                args=self.args,
            )
            preserve_mask = torch.maximum(diff_mask, roi_mask).clamp(0.0, 1.0)

            gen = torch.Generator(device=self.args.device)
            gen.manual_seed(int(self.args.seed) + scene_index * 1000 + attempt_idx * 100 + round_idx)
            with torch.no_grad():
                denoised_vectors, final_latents = self.pipeline(
                    class_labels=[map_id],
                    num_inference_timesteps=self.args.num_inference_timesteps,
                    guidance_scale=self.args.guidance_scale,
                    num_classes=self.num_classes,
                    init_latents=current_latents,
                    start_timestep_index=int(start_idx),
                    preserve_mask=preserve_mask,
                    generator=gen,
                    return_latents=True,
                )
            decoded_vector = denoised_vectors[0].torch_to_numpy(apply_sigmoid=True)
            projected_vector = apply_minimal_semantic_projection(
                candidate_vector=decoded_vector,
                anchor_vector=edited_vector,
                prompt_spec=prompt_spec,
                evaluator=self.alignment_evaluator,
                args=self.args,
            )
            projected_raster = processed_vector_to_raster(projected_vector, self.ae_config)
            current_vector = projected_vector
            current_raster = projected_raster
            current_latents = encode_raster(self.autoencoder_model, projected_raster, self.args.device)

            alignment = self.alignment_evaluator.evaluate(projected_vector, prompt_spec)
            semantic = summarize_crossing_semantics(alignment, prompt_spec, self.args.alignment_threshold)
            compliance = basic_scene_compliance(make_simulation_compatible_vector(projected_vector, edited_raw))
            round_rows.append(
                {
                    "round_idx": int(round_idx),
                    "start_timestep_index": int(start_idx),
                    "alignment_total": float(alignment.total),
                    "semantic": semantic,
                    "compliance": compliance,
                }
            )

        final_alignment = self.alignment_evaluator.evaluate(current_vector, prompt_spec)
        final_semantic = summarize_crossing_semantics(final_alignment, prompt_spec, self.args.alignment_threshold)
        final_vector = make_simulation_compatible_vector(current_vector, edited_raw)
        final_compliance = basic_scene_compliance(final_vector)
        return {
            "vector": final_vector,
            "alignment": final_alignment,
            "semantic": final_semantic,
            "compliance": final_compliance,
            "preserve_mask": preserve_mask.detach().cpu(),
            "final_latents": current_latents.detach().cpu(),
            "round_rows": round_rows,
            "used_start_step_seq": list(self.start_step_candidates),
        }

    def run_one(self, edited_scene_path: Path, out_dir: Path, index: int) -> Dict[str, object]:
        out_dir.mkdir(parents=True, exist_ok=True)
        rel = edited_scene_path.relative_to(self.edited_dir)
        original_scene_path = self.original_dir / rel
        if not original_scene_path.exists():
            raise FileNotFoundError(f"Cannot find paired original scene: {original_scene_path}")

        prompt_spec, prompt = self._load_prompt_spec(edited_scene_path.parent)
        map_id = resolve_map_id(edited_scene_path, self.args.map_id, getattr(prompt_spec, "map_id", None))

        original_raw, _ = load_raw_scene(original_scene_path)
        edited_raw, source_format = load_raw_scene(edited_scene_path)
        original_vector, original_raster = sledge_raw_feature_processing(original_raw, self.ae_config)
        edited_vector, edited_raster = sledge_raw_feature_processing(edited_raw, self.ae_config)

        edited_alignment = self.alignment_evaluator.evaluate(edited_vector, prompt_spec)
        edited_semantic = summarize_crossing_semantics(edited_alignment, prompt_spec, self.args.alignment_threshold)
        edited_sim_vector = make_simulation_compatible_vector(edited_vector, edited_raw)
        edited_compliance = basic_scene_compliance(edited_sim_vector)
        edited_total = max(float(edited_alignment.total), 1e-6)

        edit_report_path = edited_scene_path.parent / "edit_report.json"
        roi_dicts: List[Dict[str, float]] = []
        if edit_report_path.exists():
            edit_report = load_json(edit_report_path)
            roi_dicts = list(edit_report.get("preserved_rois", []))

        candidates: List[Dict[str, object]] = []
        for attempt_idx in range(max(1, int(self.args.repair_attempts))):
            candidate = self._attempt_alternating_repair(
                original_raster=original_raster,
                edited_vector=edited_vector,
                edited_raster=edited_raster,
                edited_raw=edited_raw,
                prompt_spec=prompt_spec,
                map_id=map_id,
                attempt_idx=attempt_idx,
                scene_index=index,
                roi_dicts=roi_dicts,
            )
            preservation_ratio = float(candidate["alignment"].total) / edited_total
            semantic_ok = bool(candidate["semantic"]["semantic_pass"]) and preservation_ratio >= float(self.args.min_preservation_ratio)
            compliance_ok = bool(candidate["compliance"]["compliant"])
            rank_score = 100.0 * float(semantic_ok) + 30.0 * float(compliance_ok) + 10.0 * preservation_ratio + float(candidate["alignment"].total)
            candidates.append(
                {
                    "source": f"repair_attempt_{attempt_idx:03d}",
                    **candidate,
                    "preservation_ratio": preservation_ratio,
                    "semantic_ok": semantic_ok,
                    "compliance_ok": compliance_ok,
                    "rank_score": rank_score,
                }
            )

        valid_candidates = [
            c for c in candidates
            if bool(c["semantic_ok"]) and bool(c["compliance_ok"]) and float(c["preservation_ratio"]) >= float(self.args.min_preservation_ratio)
        ]
        best = max(valid_candidates, key=lambda c: float(c["rank_score"])) if valid_candidates else None

        save_json(
            out_dir / "edited_prompt_alignment.json",
            {
                **edited_alignment.to_dict(),
                **edited_semantic,
                "prompt": prompt,
                "compliance": edited_compliance,
                "accepted": bool(edited_semantic["semantic_pass"] and edited_compliance["compliant"]),
            },
        )
        save_json(
            out_dir / "candidate_scores.json",
            [
                {
                    "source": c["source"],
                    "alignment_total": float(c["alignment"].total),
                    "semantic_summary": c["semantic"],
                    "preservation_ratio": float(c["preservation_ratio"]),
                    "compliance": c["compliance"],
                    "rank_score": float(c["rank_score"]),
                    "semantic_ok": bool(c["semantic_ok"]),
                    "compliance_ok": bool(c["compliance_ok"]),
                    "round_rows": c["round_rows"],
                    "used_start_step_seq": c["used_start_step_seq"],
                }
                for c in candidates
            ],
        )

        scenario_vector_path = None
        selected_source = None
        selected_alignment_total = None
        selected_semantic_pass = False
        selected_compliant = False
        selected_preservation_ratio = None
        used_start_step_seq: Optional[List[int]] = None

        if best is not None:
            scenario_cache_dir = self._scenario_cache_dir(edited_scene_path, index)
            scenario_cache_dir.mkdir(parents=True, exist_ok=True)
            scenario_vector_path = save_gz_pickle(
                scenario_cache_dir / "sledge_vector",
                feature_to_raw_scene_dict(best["vector"]),
            )
            selected_source = str(best["source"])
            selected_alignment_total = float(best["alignment"].total)
            selected_semantic_pass = bool(best["semantic"]["semantic_pass"])
            selected_compliant = bool(best["compliance"]["compliant"])
            selected_preservation_ratio = float(best["preservation_ratio"])
            used_start_step_seq = list(best["used_start_step_seq"])
            save_json(
                out_dir / "final_prompt_alignment.json",
                {
                    "source": selected_source,
                    "alignment_total": selected_alignment_total,
                    "semantic_summary": best["semantic"],
                    "preservation_ratio": selected_preservation_ratio,
                    "compliance": best["compliance"],
                    "used_start_step_seq": used_start_step_seq,
                    "repair_success": True,
                },
            )
            if self.args.save_latents:
                torch.save(best["final_latents"], out_dir / "best_final_latents.pt")
                torch.save(best["preserve_mask"], out_dir / "best_preserve_mask.pt")
            if self.args.save_visuals:
                save_image(out_dir / "best_vector.png", get_sledge_vector_as_raster(best["vector"], self.ae_config))
        else:
            save_json(
                out_dir / "final_prompt_alignment.json",
                {
                    "source": None,
                    "alignment_total": None,
                    "semantic_summary": None,
                    "preservation_ratio": None,
                    "compliance": None,
                    "used_start_step_seq": None,
                    "repair_success": False,
                },
            )

        if self.args.save_visuals:
            original_raster_vis = SledgeRaster(original_raster.to_feature_tensor().data.unsqueeze(0).cpu())
            edited_raster_vis = SledgeRaster(edited_raster.to_feature_tensor().data.unsqueeze(0).cpu())
            save_image(out_dir / "original_raster.png", get_sledge_raster(original_raster_vis, self.ae_config.pixel_frame))
            save_image(out_dir / "edited_raster.png", get_sledge_raster(edited_raster_vis, self.ae_config.pixel_frame))
            save_image(out_dir / "original_vector.png", get_sledge_vector_as_raster(original_vector, self.ae_config))
            save_image(out_dir / "edited_vector.png", get_sledge_vector_as_raster(edited_vector, self.ae_config))

        summary = {
            "scene_path": str(edited_scene_path),
            "original_scene_path": str(original_scene_path),
            "output_dir": str(out_dir),
            "scenario_cache_vector_path": str(scenario_vector_path) if scenario_vector_path is not None else None,
            "prompt": prompt,
            "source_format": source_format,
            "edited_alignment_total": float(edited_alignment.total),
            "edited_semantic_pass": bool(edited_semantic["semantic_pass"]),
            "edited_compliant": bool(edited_compliance["compliant"]),
            "repair_success": best is not None,
            "selected_source": selected_source,
            "selected_alignment_total": selected_alignment_total,
            "selected_semantic_pass": selected_semantic_pass,
            "selected_compliant": selected_compliant,
            "selected_preservation_ratio": selected_preservation_ratio,
            "used_start_step_seq": used_start_step_seq,
        }
        save_json(out_dir / "summary.json", summary)
        return summary

    def run_batch(self) -> None:
        total = len(self.scene_paths)
        summary_rows: List[Dict[str, object]] = []
        for index, scene_path in enumerate(self.scene_paths, start=1):
            out_dir = self._scene_output_dir(scene_path, index)
            marker = out_dir / "summary.json"
            if self.args.skip_existing and marker.exists():
                print(f"[{index}/{total}] skipped: {scene_path}")
                continue
            print(f"[{index}/{total}] processing: {scene_path}")
            try:
                row = self.run_one(scene_path, out_dir, index)
                summary_rows.append(row)
                print(
                    f"[{index}/{total}] done: {scene_path} | repair_success={row['repair_success']} | "
                    f"selected={row['selected_source']} | semantic={row['selected_semantic_pass']} | compliant={row['selected_compliant']}"
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
            "scenario_cache_root": str(self.scenario_cache_root),
            "rows": summary_rows,
        }
        save_json(self.out_root / "batch_summary.json", batch_summary)
        with open(self.out_root / "batch_summary.jsonl", "w", encoding="utf-8") as fp:
            for row in summary_rows:
                fp.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    args = build_argparser().parse_args()
    runner = HalfDenoiseFromTieredCacheRunner(args)
    runner.run_batch()


if __name__ == "__main__":
    main()
