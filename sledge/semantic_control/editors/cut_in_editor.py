from __future__ import annotations

import copy
import math
from typing import Dict, List, Tuple

import numpy as np

from sledge.autoencoder.preprocessing.features.sledge_vector_feature import (
    AgentIndex,
    SledgeVectorElement,
    SledgeVectorRaw,
)
from sledge.semantic_control.prompt_spec import PromptSpec, SceneEditResult, SceneEditROI


# V3: practical cut-in proxy for single-frame vector representation.
# Goal is not a perfect trajectory-level lane change, but a visually obvious
# adjacent-lane vehicle that cuts into ego lane ahead.
_SEVERITY_CFG: Dict[str, Dict[str, float]] = {
    "mild": {
        "start_x_min": 2.0,
        "start_x_max": 7.0,
        "cross_x_min": 9.0,
        "cross_x_max": 13.0,
        "t_center_min": 1.8,
        "t_center_max": 2.6,
        "adjacent_y": 3.2,
    },
    "moderate": {
        "start_x_min": 0.0,
        "start_x_max": 5.0,
        "cross_x_min": 6.0,
        "cross_x_max": 10.0,
        "t_center_min": 1.2,
        "t_center_max": 1.8,
        "adjacent_y": 3.3,
    },
    "aggressive": {
        "start_x_min": -2.0,
        "start_x_max": 3.0,
        "cross_x_min": 3.5,
        "cross_x_max": 7.0,
        "t_center_min": 0.8,
        "t_center_max": 1.2,
        "adjacent_y": 3.4,
    },
}


class CutInEditor:
    """Practical adjacent-lane front cut-in editor for Sledge single-frame scenes."""

    def __init__(self) -> None:
        self._ego_lane_y = 0.0
        self._lane_half_width_m = 1.8
        self._clearance_radius_m = 2.2
        self._spawn_x_jitter = [0.0, -1.0, 1.0, -2.0, 2.0]
        self._min_speed = 3.0
        self._max_speed = 15.0

    def edit(self, scene: SledgeVectorRaw, spec: PromptSpec) -> Tuple[SledgeVectorRaw, SceneEditResult]:
        edited = copy.deepcopy(scene)

        severity = getattr(spec, "severity_level", "moderate")
        if severity not in _SEVERITY_CFG:
            severity = "moderate"
        cfg = _SEVERITY_CFG[severity]

        lane_y = self._ego_lane_y
        ego_speed = self._estimate_ego_speed(edited)

        if getattr(spec, "side", "auto") == "left":
            side_sign = 1.0
        elif getattr(spec, "side", "auto") == "right":
            side_sign = -1.0
        else:
            side_sign = self._choose_side(edited)

        start_x = 0.5 * (cfg["start_x_min"] + cfg["start_x_max"])
        cross_x = 0.5 * (cfg["cross_x_min"] + cfg["cross_x_max"])
        t_center = 0.5 * (cfg["t_center_min"] + cfg["t_center_max"])
        start_y = side_sign * cfg["adjacent_y"]

        # We want the vehicle to roughly cross ego-lane center at t_center.
        # Use direct component design rather than weak heading-only drift.
        vy = -side_sign * (abs(start_y - lane_y) / max(t_center, 1e-3))
        vx = (cross_x - start_x) / max(t_center, 1e-3)

        # Ensure longitudinal motion is not too weak
        vx = max(vx, ego_speed + 0.5)
        vx = float(np.clip(vx, self._min_speed, self._max_speed))
        speed = float(np.clip(math.hypot(vx, vy), self._min_speed, self._max_speed))
        heading = float(math.atan2(vy, max(vx, 1e-3)))

        # Recompute cross_x after clipping vx for more honest notes/ROIs
        cross_x = float(start_x + vx * t_center)

        start_x, start_y = self._resolve_spawn_overlap(edited, start_x, start_y)

        # Make the insertion visually cleaner:
        # remove one same-lane vehicle near the target front-insert zone if it blocks the cut-in.
        removed_indices = self._clear_conflicting_same_lane_vehicle(
            edited.vehicles,
            target_x=cross_x,
            lane_y=lane_y,
            exclude_side_sign=side_sign,
        )

        vehicle_idx = self._allocate_vehicle_slot(edited.vehicles)
        self._set_vehicle_state(
            edited.vehicles,
            vehicle_idx,
            x=start_x,
            y=start_y,
            heading=heading,
            width=1.9,
            length=4.8,
            velocity=speed,
        )

        rois = self._build_rois(
            vehicle_state=edited.vehicles.states[vehicle_idx],
            lane_y=lane_y,
            cross_x=cross_x,
        )

        result = SceneEditResult(
            prompt_spec=spec,
            primary_actor_type="vehicle",
            primary_actor_index=vehicle_idx,
            conflict_point_xy=[float(cross_x), float(lane_y)],
            preserved_rois=rois,
            removed_vehicle_indices=removed_indices,
            notes=[
                f"scenario set to practical front cut-in ({severity})",
                f"ego speed estimate is {ego_speed:.2f} m/s",
                f"start pose = ({start_x:.2f}, {start_y:.2f})",
                f"predicted center-cross time = {t_center:.2f} s",
                f"predicted center-cross x = {cross_x:.2f} m",
                f"velocity components vx={vx:.2f} m/s vy={vy:.2f} m/s",
                f"heading={heading:.3f} rad speed={speed:.2f} m/s",
                f"side_sign={side_sign:+.0f}",
                f"removed blocking vehicles: {removed_indices}",
            ],
        )
        return edited, result

    def _estimate_ego_speed(self, scene: SledgeVectorRaw) -> float:
        ego_states = np.asarray(scene.ego.states, dtype=np.float32).reshape(-1)
        if ego_states.size == 0:
            return 6.0
        if ego_states.size >= 2:
            speed = float(np.linalg.norm(ego_states[:2]))
        else:
            speed = float(abs(ego_states[0]))
        return float(np.clip(speed, 2.5, 15.0))

    def _choose_side(self, scene: SledgeVectorRaw) -> float:
        vehicles = self._valid_agent_states(scene.vehicles)
        if len(vehicles) == 0:
            return 1.0
        left_count = int(np.sum(vehicles[:, AgentIndex.Y] > 1.5))
        right_count = int(np.sum(vehicles[:, AgentIndex.Y] < -1.5))
        return 1.0 if left_count <= right_count else -1.0

    def _resolve_spawn_overlap(self, scene: SledgeVectorRaw, target_x: float, target_y: float) -> Tuple[float, float]:
        for dx in self._spawn_x_jitter:
            cand_x = target_x + dx
            if self._is_clear(scene, cand_x, target_y):
                return cand_x, target_y
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

    def _clear_conflicting_same_lane_vehicle(
        self,
        elem: SledgeVectorElement,
        target_x: float,
        lane_y: float,
        exclude_side_sign: float,
    ) -> List[int]:
        states = np.asarray(elem.states)
        mask = np.asarray(elem.mask).astype(bool)
        removed: List[int] = []

        candidates = np.where(mask)[0]
        for idx in candidates:
            x = float(states[idx, AgentIndex.X])
            y = float(states[idx, AgentIndex.Y])

            if abs(y - lane_y) < 1.2 and (target_x - 2.0) <= x <= (target_x + 4.0):
                elem.mask[idx] = False
                removed.append(int(idx))
                break

        return removed

    def _allocate_vehicle_slot(self, elem: SledgeVectorElement) -> int:
        valid = np.asarray(elem.mask).astype(bool)
        invalid = np.where(~valid)[0]
        if len(invalid) > 0:
            idx = int(invalid[0])
            elem.mask[idx] = True
            return idx

        states = np.asarray(elem.states)
        distances = np.linalg.norm(states[:, :2], axis=1)
        idx = int(np.argmax(distances))
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

    def _build_rois(
        self,
        vehicle_state: np.ndarray,
        lane_y: float,
        cross_x: float,
    ) -> list[SceneEditROI]:
        x = float(vehicle_state[AgentIndex.X])
        y = float(vehicle_state[AgentIndex.Y])
        width = float(max(vehicle_state[AgentIndex.WIDTH], 1.8))
        length = float(max(vehicle_state[AgentIndex.LENGTH], 4.5))
        y0, y1 = sorted([y, lane_y])

        return [
            SceneEditROI(
                x_min=x - length / 2 - 1.0,
                y_min=y - width / 2 - 1.0,
                x_max=x + length / 2 + 1.0,
                y_max=y + width / 2 + 1.0,
                tag="vehicle",
            ),
            SceneEditROI(
                x_min=min(x, cross_x) - 2.0,
                y_min=y0 - 1.0,
                x_max=max(x, cross_x) + 2.0,
                y_max=y1 + 1.0,
                tag="merge_corridor",
            ),
            SceneEditROI(
                x_min=cross_x - 2.0,
                y_min=lane_y - self._lane_half_width_m - 0.8,
                x_max=cross_x + 3.0,
                y_max=lane_y + self._lane_half_width_m + 0.8,
                tag="ego_lane_entry_anchor",
            ),
        ]

    @staticmethod
    def _valid_agent_states(elem: SledgeVectorElement) -> np.ndarray:
        valid = np.asarray(elem.mask).astype(bool)
        states = np.asarray(elem.states)
        return states[valid] if np.any(valid) else np.zeros((0, states.shape[-1]), dtype=states.dtype)