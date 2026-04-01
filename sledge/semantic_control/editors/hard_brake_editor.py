from __future__ import annotations

import copy
from typing import Dict, Optional, Tuple

import numpy as np

from sledge.autoencoder.preprocessing.features.sledge_vector_feature import (
    AgentIndex,
    SledgeVectorElement,
    SledgeVectorRaw,
)
from sledge.semantic_control.prompt_spec import PromptSpec, SceneEditResult, SceneEditROI


_SEVERITY_CFG: Dict[str, Dict[str, float]] = {
    "mild": {
        "ttc_min": 2.8,
        "ttc_max": 4.0,
        "lead_speed_ratio": 0.62,
        "lead_dist_min": 12.0,
        "lead_dist_max": 20.0,
    },
    "moderate": {
        "ttc_min": 1.9,
        "ttc_max": 3.0,
        "lead_speed_ratio": 0.42,
        "lead_dist_min": 8.0,
        "lead_dist_max": 14.0,
    },
    "aggressive": {
        "ttc_min": 1.1,
        "ttc_max": 2.0,
        "lead_speed_ratio": 0.22,
        "lead_dist_min": 5.0,
        "lead_dist_max": 10.0,
    },
}


class HardBrakeEditor:
    """
    Lead-vehicle hard-brake proxy editor.

    Main changes versus the previous version:
      1) Prefer an existing forward same-lane lead vehicle before inserting.
      2) Use TTC-consistent target distance instead of a fixed midpoint only.
      3) Use ego-lane-centered placement (y ~= 0) to better align with evaluator.
    """

    def __init__(self) -> None:
        self._lane_half_width_m = 1.8
        self._same_lane_thresh_m = 1.35
        self._clearance_radius_m = 2.5
        self._x_jitter_candidates = [0.0, 1.0, -1.0, 2.0, -2.0, 3.5]

    def edit(self, scene: SledgeVectorRaw, spec: PromptSpec) -> Tuple[SledgeVectorRaw, SceneEditResult]:
        edited = copy.deepcopy(scene)
        severity = getattr(spec, "severity_level", "moderate")
        if severity not in _SEVERITY_CFG:
            severity = "moderate"
        cfg = _SEVERITY_CFG[severity]

        ego_speed = self._estimate_ego_speed(edited)
        lane_y = self._estimate_ego_lane_y(edited)

        ttc_min = float(getattr(spec, "ttc_min_s", cfg["ttc_min"]))
        ttc_max = float(getattr(spec, "ttc_max_s", cfg["ttc_max"]))
        target_ttc = 0.5 * (ttc_min + ttc_max)

        lead_dist_min = float(getattr(spec, "lead_distance_min_m", cfg["lead_dist_min"]))
        lead_dist_max = float(getattr(spec, "lead_distance_max_m", cfg["lead_dist_max"]))

        lead_speed_ratio = float(cfg["lead_speed_ratio"])
        target_lead_speed = float(np.clip(ego_speed * lead_speed_ratio, 0.2, max(1.2, ego_speed)))

        relative_closure = max(ego_speed - target_lead_speed, 0.8)
        ttc_implied_dist = target_ttc * relative_closure
        target_x = float(np.clip(ttc_implied_dist, lead_dist_min, lead_dist_max))
        target_x = float(np.clip(target_x, 4.0, 28.0))
        target_y = float(lane_y)

        existing_idx = self._find_best_existing_lead(edited.vehicles, target_x, lane_y)
        if existing_idx is not None:
            lead_idx = existing_idx
            resolved_x, resolved_y = self._resolve_overlap(
                edited, target_x, target_y, ignore_vehicle_index=lead_idx
            )
        else:
            resolved_x, resolved_y = self._resolve_overlap(edited, target_x, target_y, ignore_vehicle_index=None)
            lead_idx = self._select_or_allocate_vehicle(edited.vehicles, resolved_x, resolved_y, lane_y)

        self._set_vehicle_state(
            edited.vehicles,
            lead_idx,
            x=resolved_x,
            y=resolved_y,
            heading=0.0,
            width=1.9,
            length=4.8,
            velocity=target_lead_speed,
        )

        actual_relative_closure = max(ego_speed - target_lead_speed, 1e-3)
        actual_ttc = float(resolved_x / actual_relative_closure)

        rois = self._build_rois(edited.vehicles.states[lead_idx], lane_y)
        result = SceneEditResult(
            prompt_spec=spec,
            primary_actor_type="lead_vehicle",
            primary_actor_index=lead_idx,
            conflict_point_xy=[float(resolved_x), float(resolved_y)],
            preserved_rois=rois,
            slowed_vehicle_indices=[lead_idx],
            notes=[
                f"scenario set to hard-brake lead vehicle ({severity})",
                f"ego speed estimate is {ego_speed:.2f} m/s",
                f"requested TTC target is {target_ttc:.2f} s",
                f"lead vehicle distance set to {resolved_x:.2f} m",
                f"lead vehicle speed set to {target_lead_speed:.2f} m/s",
                f"realized TTC is approximately {actual_ttc:.2f} s",
                "existing forward same-lane lead was reused when available",
            ],
        )
        return edited, result

    def _estimate_ego_speed(self, scene: SledgeVectorRaw) -> float:
        ego_states = np.asarray(scene.ego.states).reshape(-1)
        if ego_states.size == 0:
            return 6.0
        speed = float(abs(ego_states[0]))
        return float(np.clip(speed, 2.5, 15.0))

    def _estimate_ego_lane_y(self, scene: SledgeVectorRaw) -> float:
        vehicles = self._valid_agent_states(scene.vehicles)
        if len(vehicles) == 0:
            return 0.0
        same_band = vehicles[np.abs(vehicles[:, AgentIndex.Y]) < 2.2]
        if len(same_band) == 0:
            return 0.0
        ahead = same_band[(same_band[:, AgentIndex.X] > -5.0) & (same_band[:, AgentIndex.X] < 35.0)]
        if len(ahead) == 0:
            return 0.0
        return float(np.clip(np.median(ahead[:, AgentIndex.Y]), -0.8, 0.8))

    def _find_best_existing_lead(
        self,
        elem: SledgeVectorElement,
        target_x: float,
        lane_y: float,
    ) -> Optional[int]:
        states = np.asarray(elem.states)
        valid = np.asarray(elem.mask).astype(bool)
        if not np.any(valid):
            return None

        candidates = np.where(valid)[0]
        best_idx: Optional[int] = None
        best_cost = float("inf")
        for idx in candidates:
            x = float(states[idx, AgentIndex.X])
            y = float(states[idx, AgentIndex.Y])
            if x <= 1.5:
                continue
            lateral = abs(y - lane_y)
            if lateral > self._same_lane_thresh_m:
                continue
            cost = abs(x - target_x) + 2.5 * lateral
            if cost < best_cost:
                best_cost = cost
                best_idx = int(idx)
        return best_idx

    def _resolve_overlap(
        self,
        scene: SledgeVectorRaw,
        x: float,
        y: float,
        ignore_vehicle_index: Optional[int],
    ) -> Tuple[float, float]:
        for dx in self._x_jitter_candidates:
            cand_x = x + dx
            if self._is_clear(scene, cand_x, y, ignore_vehicle_index=ignore_vehicle_index):
                return float(cand_x), float(y)
        return float(x), float(y)

    def _is_clear(
        self,
        scene: SledgeVectorRaw,
        x: float,
        y: float,
        ignore_vehicle_index: Optional[int],
    ) -> bool:
        p = np.array([x, y], dtype=np.float32)

        for elem_name, elem in [
            ("vehicles", scene.vehicles),
            ("pedestrians", scene.pedestrians),
            ("static_objects", scene.static_objects),
        ]:
            valid = np.asarray(elem.mask).astype(bool)
            if not np.any(valid):
                continue
            states = np.asarray(elem.states)
            idxs = np.where(valid)[0]
            for idx in idxs:
                if elem_name == "vehicles" and ignore_vehicle_index is not None and int(idx) == int(ignore_vehicle_index):
                    continue
                center = np.asarray(states[idx, :2], dtype=np.float32)
                dist = float(np.linalg.norm(center - p))
                if dist < self._clearance_radius_m:
                    return False
        return True

    def _select_or_allocate_vehicle(self, elem: SledgeVectorElement, x: float, y: float, lane_y: float) -> int:
        states = np.asarray(elem.states)
        valid = np.asarray(elem.mask).astype(bool)

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
            x_min=max(0.0, x - 5.0),
            y_min=lane_y - self._lane_half_width_m - 0.8,
            x_max=x + 5.0,
            y_max=lane_y + self._lane_half_width_m + 0.8,
            tag="longitudinal_conflict_anchor",
        )
        stopping_roi = SceneEditROI(
            x_min=x - 2.5,
            y_min=lane_y - self._lane_half_width_m - 0.5,
            x_max=x + 2.5,
            y_max=lane_y + self._lane_half_width_m + 0.5,
            tag="lead_brake_zone",
        )
        return [lead_roi, lane_conflict_roi, stopping_roi]

    @staticmethod
    def _valid_agent_states(elem: SledgeVectorElement) -> np.ndarray:
        valid = np.asarray(elem.mask).astype(bool)
        states = np.asarray(elem.states)
        return states[valid] if np.any(valid) else np.zeros((0, states.shape[-1]), dtype=states.dtype)
