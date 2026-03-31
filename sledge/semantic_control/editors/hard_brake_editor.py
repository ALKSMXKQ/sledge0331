from __future__ import annotations

import copy
import math
from typing import Dict, Tuple

import numpy as np

from sledge.autoencoder.preprocessing.features.sledge_vector_feature import AgentIndex, SledgeVectorElement, SledgeVectorRaw
from sledge.semantic_control.prompt_spec import PromptSpec, SceneEditResult, SceneEditROI


_SEVERITY_CFG: Dict[str, Dict[str, float]] = {
    "mild": {"lead_dist_min": 12.0, "lead_dist_max": 20.0, "lead_speed_ratio": 0.65},
    "moderate": {"lead_dist_min": 8.0, "lead_dist_max": 14.0, "lead_speed_ratio": 0.45},
    "aggressive": {"lead_dist_min": 5.0, "lead_dist_max": 10.0, "lead_speed_ratio": 0.20},
}


class HardBrakeEditor:
    """Lead-vehicle hard-brake proxy editor using short-gap low-speed lead placement."""

    def __init__(self) -> None:
        self._lane_half_width_m = 1.8
        self._clearance_radius_m = 2.5

    def edit(self, scene: SledgeVectorRaw, spec: PromptSpec) -> Tuple[SledgeVectorRaw, SceneEditResult]:
        edited = copy.deepcopy(scene)
        severity = getattr(spec, "severity_level", "moderate")
        if severity not in _SEVERITY_CFG:
            severity = "moderate"
        cfg = _SEVERITY_CFG[severity]

        ego_speed = self._estimate_ego_speed(edited)
        lane_y = self._estimate_lane_center_y(edited)

        lead_dist_min = float(getattr(spec, "lead_distance_min_m", cfg["lead_dist_min"]))
        lead_dist_max = float(getattr(spec, "lead_distance_max_m", cfg["lead_dist_max"]))
        lead_dist = 0.5 * (lead_dist_min + lead_dist_max)

        lead_speed = float(np.clip(ego_speed * cfg["lead_speed_ratio"], 0.2, max(1.0, ego_speed)))
        target_x = float(np.clip(lead_dist, 4.0, 28.0))
        target_y = lane_y
        target_x, target_y = self._resolve_overlap(edited, target_x, target_y)

        lead_idx = self._select_or_allocate_vehicle(edited.vehicles, target_x, target_y, lane_y)
        self._set_vehicle_state(
            edited.vehicles,
            lead_idx,
            x=target_x,
            y=target_y,
            heading=0.0,
            width=1.9,
            length=4.8,
            velocity=lead_speed,
        )

        rois = self._build_rois(edited.vehicles.states[lead_idx], lane_y)
        result = SceneEditResult(
            prompt_spec=spec,
            primary_actor_type="lead_vehicle",
            primary_actor_index=lead_idx,
            conflict_point_xy=[float(target_x), float(target_y)],
            preserved_rois=rois,
            slowed_vehicle_indices=[lead_idx],
            notes=[
                f"scenario set to hard-brake lead vehicle ({severity})",
                f"ego speed estimate is {ego_speed:.2f} m/s",
                f"lead vehicle distance set to {lead_dist:.2f} m",
                f"lead vehicle speed set to {lead_speed:.2f} m/s",
                "hard brake is represented as a short-gap, low-speed lead vehicle proxy",
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

    def _resolve_overlap(self, scene: SledgeVectorRaw, x: float, y: float) -> Tuple[float, float]:
        for dx in [0.0, 1.5, -1.5, 3.0]:
            cand_x = x + dx
            if self._is_clear(scene, cand_x, y):
                return cand_x, y
        return x, y

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

    def _select_or_allocate_vehicle(self, elem: SledgeVectorElement, x: float, y: float, lane_y: float) -> int:
        states = np.asarray(elem.states)
        valid = np.asarray(elem.mask).astype(bool)
        candidates = np.where(valid)[0]
        if len(candidates) > 0:
            same_lane = [idx for idx in candidates if abs(states[idx, AgentIndex.Y] - lane_y) < 1.5 and states[idx, AgentIndex.X] > 0.0]
            if len(same_lane) > 0:
                same_lane = sorted(same_lane, key=lambda idx: abs(states[idx, AgentIndex.X] - x))
                return int(same_lane[0])

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

    def _build_rois(self, lead_state: np.ndarray, lane_y: float) -> list[SceneEditROI]:
        x = float(lead_state[AgentIndex.X])
        y = float(lead_state[AgentIndex.Y])
        width = float(max(lead_state[AgentIndex.WIDTH], 1.8))
        length = float(max(lead_state[AgentIndex.LENGTH], 4.5))
        lead_roi = SceneEditROI(
            x_min=x - length / 2 - 1.0,
            y_min=y - width / 2 - 1.0,
            x_max=x + length / 2 + 1.0,
            y_max=y + width / 2 + 1.0,
            tag="lead_vehicle",
        )
        lane_conflict_roi = SceneEditROI(
            x_min=max(0.0, x - 4.0),
            y_min=lane_y - self._lane_half_width_m - 0.8,
            x_max=x + 4.0,
            y_max=lane_y + self._lane_half_width_m + 0.8,
            tag="longitudinal_conflict_anchor",
        )
        stopping_roi = SceneEditROI(
            x_min=x - 2.0,
            y_min=lane_y - self._lane_half_width_m - 0.5,
            x_max=x + 2.0,
            y_max=lane_y + self._lane_half_width_m + 0.5,
            tag="lead_brake_zone",
        )
        return [lead_roi, lane_conflict_roi, stopping_roi]

    @staticmethod
    def _valid_agent_states(elem: SledgeVectorElement) -> np.ndarray:
        valid = np.asarray(elem.mask).astype(bool)
        states = np.asarray(elem.states)
        return states[valid] if np.any(valid) else np.zeros((0, states.shape[-1]), dtype=states.dtype)
