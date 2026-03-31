from __future__ import annotations

import copy
import math
from typing import Dict, Tuple

import numpy as np

from sledge.autoencoder.preprocessing.features.sledge_vector_feature import AgentIndex, SledgeVectorElement, SledgeVectorRaw
from sledge.semantic_control.prompt_spec import PromptSpec, SceneEditResult, SceneEditROI


_SEVERITY_CFG: Dict[str, Dict[str, float]] = {
    "mild": {"gap_min": 10.0, "gap_max": 16.0, "lat_speed_min": 0.8, "lat_speed_max": 1.3, "rel_speed_min": 1.5, "rel_speed_max": 3.0},
    "moderate": {"gap_min": 6.0, "gap_max": 10.0, "lat_speed_min": 1.0, "lat_speed_max": 1.8, "rel_speed_min": 2.5, "rel_speed_max": 4.5},
    "aggressive": {"gap_min": 3.5, "gap_max": 7.0, "lat_speed_min": 1.3, "lat_speed_max": 2.4, "rel_speed_min": 3.5, "rel_speed_max": 6.0},
}


class CutInEditor:
    """Adjacent-lane vehicle sudden cut-in editor."""

    def __init__(self) -> None:
        self._lane_half_width_m = 1.8
        self._adjacent_lane_offset_m = 3.6
        self._clearance_radius_m = 2.0
        self._x_jitter_candidates = [0.0, -1.0, 1.0, -2.0, 2.0]

    def edit(self, scene: SledgeVectorRaw, spec: PromptSpec) -> Tuple[SledgeVectorRaw, SceneEditResult]:
        edited = copy.deepcopy(scene)
        severity = getattr(spec, "severity_level", "moderate")
        if severity not in _SEVERITY_CFG:
            severity = "moderate"
        cfg = _SEVERITY_CFG[severity]

        ego_speed = self._estimate_ego_speed(edited)
        lane_y = self._estimate_lane_center_y(edited)

        gap_min = float(getattr(spec, "target_gap_min_m", cfg["gap_min"]))
        gap_max = float(getattr(spec, "target_gap_max_m", cfg["gap_max"]))
        target_gap = 0.5 * (gap_min + gap_max)

        rel_speed = 0.5 * (cfg["rel_speed_min"] + cfg["rel_speed_max"])
        lat_speed = 0.5 * (cfg["lat_speed_min"] + cfg["lat_speed_max"])
        if getattr(spec, "side", "auto") == "left":
            side_sign = 1.0
        elif getattr(spec, "side", "auto") == "right":
            side_sign = -1.0
        else:
            side_sign = self._choose_side(edited, lane_y)

        target_x = float(np.clip(target_gap, 4.0, 22.0))
        target_y = lane_y + side_sign * self._adjacent_lane_offset_m

        target_x, target_y = self._resolve_spawn_overlap(edited, target_x, target_y, lane_y, side_sign)
        heading = -side_sign * math.atan2(lat_speed, max(ego_speed + rel_speed, 1e-3))

        vehicle_idx = self._select_or_allocate_vehicle(edited.vehicles, target_x, target_y, lane_y, side_sign)
        merge_speed = float(np.clip(ego_speed + rel_speed, 2.0, 15.0))
        self._set_vehicle_state(
            edited.vehicles,
            vehicle_idx,
            x=target_x,
            y=target_y,
            heading=heading,
            width=1.9,
            length=4.8,
            velocity=merge_speed,
        )

        conflict_point = [float(target_x + 2.0), float(lane_y)]
        rois = self._build_rois(edited.vehicles.states[vehicle_idx], lane_y, side_sign)

        result = SceneEditResult(
            prompt_spec=spec,
            primary_actor_type="vehicle",
            primary_actor_index=vehicle_idx,
            conflict_point_xy=conflict_point,
            preserved_rois=rois,
            notes=[
                f"scenario set to vehicle cut-in ({severity})",
                f"ego speed estimate is {ego_speed:.2f} m/s",
                f"target merge gap is {target_gap:.2f} m",
                f"merge vehicle speed set to {merge_speed:.2f} m/s",
                f"lateral merge direction side_sign={side_sign:+.0f}",
            ],
        )
        return edited, result

    def _estimate_ego_speed(self, scene: SledgeVectorRaw) -> float:
        ego_states = np.asarray(scene.ego.states).reshape(-1)
        if ego_states.size == 0:
            return 6.0
        speed = float(abs(ego_states[0]))
        return float(np.clip(speed, 2.5, 15.0))

    def _estimate_lane_center_y(self, scene: SledgeVectorRaw) -> float:
        vehicles = self._valid_agent_states(scene.vehicles)
        if len(vehicles) == 0:
            return 0.0
        forward = vehicles[(vehicles[:, AgentIndex.X] > 2.0) & (vehicles[:, AgentIndex.X] < 30.0)]
        if len(forward) == 0:
            return 0.0
        usable = forward[np.abs(forward[:, AgentIndex.Y]) < 4.0]
        target = usable if len(usable) > 0 else forward
        return float(np.median(target[:, AgentIndex.Y]))

    def _choose_side(self, scene: SledgeVectorRaw, lane_y: float) -> float:
        vehicles = self._valid_agent_states(scene.vehicles)
        if len(vehicles) == 0:
            return 1.0
        left_count = int(np.sum(vehicles[:, AgentIndex.Y] > lane_y + 1.5))
        right_count = int(np.sum(vehicles[:, AgentIndex.Y] < lane_y - 1.5))
        return 1.0 if left_count <= right_count else -1.0

    def _resolve_spawn_overlap(self, scene: SledgeVectorRaw, target_x: float, target_y: float, lane_y: float, side_sign: float) -> Tuple[float, float]:
        for dx in self._x_jitter_candidates:
            cand_x = target_x + dx
            cand_y = target_y
            if self._is_clear(scene, cand_x, cand_y):
                return cand_x, cand_y
        return target_x, target_y

    def _is_clear(self, scene: SledgeVectorRaw, x: float, y: float) -> bool:
        p = np.array([x, y], dtype=np.float32)
        for elem in [scene.vehicles, scene.pedestrians, scene.static_objects]:
            valid = np.asarray(elem.mask).astype(bool)
            if not np.any(valid):
                continue
            centers = np.asarray(elem.states)[valid, :2]
            if centers.size == 0:
                continue
            dists = np.linalg.norm(centers - p[None, :], axis=1)
            if np.any(dists < self._clearance_radius_m):
                return False
        return True

    def _select_or_allocate_vehicle(self, elem: SledgeVectorElement, x: float, y: float, lane_y: float, side_sign: float) -> int:
        states = np.asarray(elem.states)
        valid = np.asarray(elem.mask).astype(bool)
        candidates = np.where(valid)[0]
        if len(candidates) > 0:
            adj = [
                idx for idx in candidates
                if (states[idx, AgentIndex.Y] - lane_y) * side_sign > 1.5 and abs(states[idx, AgentIndex.X] - x) < 12.0
            ]
            if len(adj) > 0:
                return int(adj[0])

        invalid = np.where(~valid)[0]
        if len(invalid) > 0:
            idx = int(invalid[0])
            elem.mask[idx] = True
            return idx

        distances = np.linalg.norm(states[:, :2] - np.array([[x, y]], dtype=np.float32), axis=1)
        idx = int(np.argmin(distances))
        elem.mask[idx] = True
        return idx

    @staticmethod
    def _set_vehicle_state(
        elem: SledgeVectorElement,
        idx: int,
        x: float,
        y: float,
        heading: float,
        width: float,
        length: float,
        velocity: float,
    ) -> None:
        elem.states[idx, AgentIndex.X] = x
        elem.states[idx, AgentIndex.Y] = y
        elem.states[idx, AgentIndex.HEADING] = heading
        elem.states[idx, AgentIndex.WIDTH] = width
        elem.states[idx, AgentIndex.LENGTH] = length
        elem.states[idx, AgentIndex.VELOCITY] = velocity
        elem.mask[idx] = True

    def _build_rois(self, vehicle_state: np.ndarray, lane_y: float, side_sign: float) -> list[SceneEditROI]:
        x = float(vehicle_state[AgentIndex.X])
        y = float(vehicle_state[AgentIndex.Y])
        width = float(max(vehicle_state[AgentIndex.WIDTH], 1.8))
        length = float(max(vehicle_state[AgentIndex.LENGTH], 4.5))
        merge_target_y = lane_y

        vehicle_roi = SceneEditROI(
            x_min=x - length / 2 - 1.0,
            y_min=y - width / 2 - 1.0,
            x_max=x + length / 2 + 1.0,
            y_max=y + width / 2 + 1.0,
            tag="vehicle",
        )
        y0, y1 = sorted([y, merge_target_y])
        corridor_roi = SceneEditROI(
            x_min=x - 2.0,
            y_min=y0 - 1.0,
            x_max=x + 6.0,
            y_max=y1 + 1.0,
            tag="merge_corridor",
        )
        target_lane_roi = SceneEditROI(
            x_min=x - 1.0,
            y_min=lane_y - self._lane_half_width_m - 0.8,
            x_max=x + 6.0,
            y_max=lane_y + self._lane_half_width_m + 0.8,
            tag="ego_lane_conflict_anchor",
        )
        adjacent_anchor_roi = SceneEditROI(
            x_min=x - 1.5,
            y_min=y - 1.0,
            x_max=x + 1.5,
            y_max=y + 1.0,
            tag="adjacent_lane_spawn_anchor",
        )
        return [vehicle_roi, corridor_roi, target_lane_roi, adjacent_anchor_roi]

    @staticmethod
    def _valid_agent_states(elem: SledgeVectorElement) -> np.ndarray:
        valid = np.asarray(elem.mask).astype(bool)
        states = np.asarray(elem.states)
        return states[valid] if np.any(valid) else np.zeros((0, states.shape[-1]), dtype=states.dtype)
