from __future__ import annotations

import copy
import math
from typing import Dict, Tuple

import numpy as np

from sledge.autoencoder.preprocessing.features.sledge_vector_feature import (
    AgentIndex,
    SledgeVectorElement,
    SledgeVectorRaw,
)
from sledge.semantic_control.prompt_spec import PromptSpec, SceneEditResult, SceneEditROI


# TTC tiers for the new "sudden pedestrian crossing" semantics.
# The key difference from earlier versions is that TTC is solved explicitly:
#   1) choose target TTC in band
#   2) choose pedestrian speed in band
#   3) back-solve roadside spawn distance so pedestrian enters lane at target TTC
#   4) back-solve forward conflict x so ego reaches that x at target TTC
_SEVERITY_CFG: Dict[str, Dict[str, float]] = {
    "mild": {
        "ttc_min_s": 3.0,
        "ttc_max_s": 4.2,
        "ped_speed_min": 1.1,
        "ped_speed_max": 1.5,
        "conflict_x_min": 12.0,
        "conflict_x_max": 22.0,
        "roadside_extra_min": 2.8,
        "roadside_extra_max": 6.5,
    },
    "moderate": {
        "ttc_min_s": 2.0,
        "ttc_max_s": 3.0,
        "ped_speed_min": 1.3,
        "ped_speed_max": 1.8,
        "conflict_x_min": 9.0,
        "conflict_x_max": 18.0,
        "roadside_extra_min": 2.2,
        "roadside_extra_max": 5.0,
    },
    "aggressive": {
        "ttc_min_s": 1.2,
        "ttc_max_s": 2.0,
        "ped_speed_min": 1.6,
        "ped_speed_max": 2.3,
        "conflict_x_min": 7.0,
        "conflict_x_max": 14.0,
        "roadside_extra_min": 1.4,
        "roadside_extra_max": 4.0,
    },
}


class SemanticSceneEditor:
    """
    Strict TTC-inversion editor for:
        "突发的行人横穿马路"

    Compared with the previous tiered editor, this version enforces the tier by geometry:
      - target TTC is solved first
      - pedestrian spawn distance from lane edge is solved from TTC and pedestrian speed
      - forward conflict x is solved from TTC and ego speed

    Editing principles:
      1) Insert or reposition only ONE pedestrian.
      2) Do not prune traffic / do not add occluders / do not modify unrelated actors.
      3) Spawn from roadside / lane edge outside the drivable lane.
      4) Use explicit TTC inversion so mild / moderate / aggressive produce distinct timing.
      5) Keep preserve ROIs local and compatible with downstream mask logic.
    """

    def __init__(self) -> None:
        self._lane_half_width_m = 1.8
        self._spawn_margin_m = 0.4
        self._clearance_radius_m = 1.4
        self._x_jitter_candidates = [0.0, -0.6, 0.6, -1.2, 1.2]

    def edit(self, scene: SledgeVectorRaw, spec: PromptSpec) -> Tuple[SledgeVectorRaw, SceneEditResult]:
        edited = self._deepcopy_scene(scene)

        if self._looks_like_pedestrian_crossing(spec):
            return self._edit_strict_ttc_crossing(edited, spec)

        result = SceneEditResult(
            prompt_spec=spec,
            occluder_source="none",
            occluder_index=-1,
            pedestrian_index=-1,
            conflict_point_xy=[float(spec.conflict_point_x_m), float(spec.conflict_point_y_m)],
            notes=["No pedestrian-crossing rule matched; scene left unchanged."],
        )
        return edited, result

    def _looks_like_pedestrian_crossing(self, spec: PromptSpec) -> bool:
        text = f"{spec.raw_prompt} {spec.normalized_prompt} {spec.scenario_type}".lower()
        keys = ["横穿", "过马路", "crossing", "cross road", "pedestrian crossing", "jaywalk", "行人", "pedestrian"]
        return any(k in text for k in keys)

    def _edit_strict_ttc_crossing(self, scene: SledgeVectorRaw, spec: PromptSpec) -> Tuple[SledgeVectorRaw, SceneEditResult]:
        severity = getattr(spec, "severity_level", "moderate")
        if severity not in _SEVERITY_CFG:
            severity = "moderate"
        cfg = _SEVERITY_CFG[severity]

        lane_y = self._estimate_lane_center_y(scene)
        ego_speed = self._estimate_ego_speed(scene)

        ttc_min = float(getattr(spec, "ttc_min_s", cfg["ttc_min_s"]))
        ttc_max = float(getattr(spec, "ttc_max_s", cfg["ttc_max_s"]))
        target_ttc = self._choose_target_ttc(ttc_min, ttc_max)

        # Pick a realistic pedestrian speed for the requested tier.
        prompt_speed = float(getattr(spec, "pedestrian_speed", 0.5 * (cfg["ped_speed_min"] + cfg["ped_speed_max"])))
        ped_speed = float(np.clip(prompt_speed, cfg["ped_speed_min"], cfg["ped_speed_max"]))

        # Solve conflict x from ego TTC: x_conflict ≈ v_ego * ttc.
        conflict_x = float(np.clip(ego_speed * target_ttc, cfg["conflict_x_min"], cfg["conflict_x_max"]))
        if abs(float(getattr(spec, "conflict_point_x_m", 12.0)) - 12.0) > 1e-3:
            conflict_x = float(np.clip(float(spec.conflict_point_x_m), cfg["conflict_x_min"], cfg["conflict_x_max"]))

        # Solve roadside spawn distance from lane edge so lane entry time matches TTC.
        roadside_extra = float(np.clip(ped_speed * target_ttc, cfg["roadside_extra_min"], cfg["roadside_extra_max"]))

        side_sign = self._choose_spawn_side(scene, conflict_x, lane_y, roadside_extra, spec)

        # Start from roadside / curbside outside the lane.
        start_y = lane_y + side_sign * (self._lane_half_width_m + roadside_extra)
        # The pedestrian crosses through the lane and exits the other side.
        end_y = lane_y - side_sign * (self._lane_half_width_m + 0.8)

        start_x = conflict_x
        start_x, start_y = self._resolve_spawn_overlap(scene, start_x, start_y, conflict_x, lane_y, side_sign, roadside_extra)

        # Recompute effective TTC after overlap resolution so summary is honest.
        effective_outside_gap = max(0.0, abs(start_y - lane_y) - self._lane_half_width_m)
        effective_ttc = effective_outside_gap / max(ped_speed, 1e-3)

        heading = -side_sign * math.pi / 2.0
        ped_idx = self._allocate_slot(scene.pedestrians)
        self._set_agent_state(
            scene.pedestrians,
            ped_idx,
            x=start_x,
            y=start_y,
            heading=heading,
            width=0.75,
            length=0.75,
            velocity=ped_speed,
        )

        conflict_point_xy = [float(start_x), float(lane_y)]
        preserved_rois = self._build_crossing_rois(
            ped_state=scene.pedestrians.states[ped_idx],
            lane_y=lane_y,
            end_y=end_y,
            conflict_x=start_x,
            roadside_extra=effective_outside_gap,
        )

        result = SceneEditResult(
            prompt_spec=spec,
            occluder_source="none",
            occluder_index=-1,
            pedestrian_index=ped_idx,
            conflict_point_xy=conflict_point_xy,
            preserved_rois=preserved_rois,
            removed_vehicle_indices=[],
            slowed_vehicle_indices=[],
            notes=[
                f"scenario set to strict TTC pedestrian crossing ({severity})",
                "only one pedestrian is inserted / repositioned",
                "no unrelated actors are modified",
                f"ego speed estimate is {ego_speed:.2f} m/s",
                f"target TTC band is [{ttc_min:.2f}, {ttc_max:.2f}] s",
                f"target TTC chosen as {target_ttc:.2f} s",
                f"pedestrian speed set to {ped_speed:.2f} m/s",
                f"roadside spawn extra distance solved as {effective_outside_gap:.2f} m",
                f"effective lane-entry TTC is approximately {effective_ttc:.2f} s",
                f"conflict point solved near x={start_x:.2f}, lane_y={lane_y:.2f}",
            ],
        )
        return scene, result

    def _choose_target_ttc(self, ttc_min: float, ttc_max: float) -> float:
        # Deterministic midpoint keeps runs reproducible.
        return float(np.clip(0.5 * (ttc_min + ttc_max), ttc_min, ttc_max))

    def _estimate_lane_center_y(self, scene: SledgeVectorRaw) -> float:
        vehicles = self._valid_agent_states(scene.vehicles)
        if len(vehicles) == 0:
            return 0.0
        forward = vehicles[(vehicles[:, AgentIndex.X] > 3.0) & (vehicles[:, AgentIndex.X] < 28.0)]
        if len(forward) == 0:
            return 0.0
        usable = forward[np.abs(forward[:, AgentIndex.Y]) < 4.5]
        target = usable if len(usable) > 0 else forward
        return float(np.median(target[:, AgentIndex.Y]))

    def _estimate_ego_speed(self, scene: SledgeVectorRaw) -> float:
        ego_states = np.asarray(scene.ego.states).reshape(-1)
        if ego_states.size == 0:
            return 6.0
        speed = float(abs(ego_states[0]))
        return float(np.clip(speed, 2.5, 15.0))

    def _choose_spawn_side(
        self,
        scene: SledgeVectorRaw,
        conflict_x: float,
        lane_y: float,
        roadside_extra: float,
        spec: PromptSpec,
    ) -> float:
        if getattr(spec, "side", "auto") == "left":
            return 1.0
        if getattr(spec, "side", "auto") == "right":
            return -1.0

        left_y = lane_y + self._lane_half_width_m + roadside_extra
        right_y = lane_y - self._lane_half_width_m - roadside_extra
        left_score = self._local_density(scene, conflict_x, left_y)
        right_score = self._local_density(scene, conflict_x, right_y)
        return 1.0 if left_score <= right_score else -1.0

    def _resolve_spawn_overlap(
        self,
        scene: SledgeVectorRaw,
        x: float,
        y: float,
        conflict_x: float,
        lane_y: float,
        side_sign: float,
        roadside_extra: float,
    ) -> Tuple[float, float]:
        base_y = lane_y + side_sign * (self._lane_half_width_m + roadside_extra)
        for dx in self._x_jitter_candidates:
            cand_x = conflict_x + dx
            cand_y = base_y
            if self._is_spawn_clear(scene, cand_x, cand_y):
                return float(cand_x), float(cand_y)

        # Fallback: move slightly further to roadside but keep TTC geometry close.
        for dy_extra in [0.5, 1.0, 1.5]:
            cand_y = lane_y + side_sign * (self._lane_half_width_m + roadside_extra + dy_extra)
            for dx in self._x_jitter_candidates:
                cand_x = conflict_x + dx
                if self._is_spawn_clear(scene, cand_x, cand_y):
                    return float(cand_x), float(cand_y)

        return float(x), float(y)

    def _is_spawn_clear(self, scene: SledgeVectorRaw, x: float, y: float) -> bool:
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

    def _local_density(self, scene: SledgeVectorRaw, x: float, y: float) -> float:
        p = np.array([x, y], dtype=np.float32)
        score = 0.0
        for elem in [scene.vehicles, scene.pedestrians, scene.static_objects]:
            valid = np.asarray(elem.mask).astype(bool)
            if not np.any(valid):
                continue
            centers = np.asarray(elem.states)[valid, :2]
            if centers.size == 0:
                continue
            dists = np.linalg.norm(centers - p[None, :], axis=1)
            score += float(np.sum(np.exp(-0.5 * (dists / 3.0) ** 2)))
        return score

    def _build_crossing_rois(
        self,
        ped_state: np.ndarray,
        lane_y: float,
        end_y: float,
        conflict_x: float,
        roadside_extra: float,
    ) -> list[SceneEditROI]:
        x = float(ped_state[AgentIndex.X])
        y = float(ped_state[AgentIndex.Y])
        width = float(max(ped_state[AgentIndex.WIDTH], 0.75))
        length = float(max(ped_state[AgentIndex.LENGTH], 0.75))

        ped_roi = SceneEditROI(
            x_min=x - length / 2 - 1.0,
            y_min=y - width / 2 - 1.0,
            x_max=x + length / 2 + 1.0,
            y_max=y + width / 2 + 1.0,
            tag="pedestrian",
        )

        y0, y1 = sorted([y, end_y])
        corridor_half_x = 1.2
        corridor_roi = SceneEditROI(
            x_min=conflict_x - corridor_half_x,
            y_min=y0 - 0.6,
            x_max=conflict_x + corridor_half_x,
            y_max=y1 + 0.6,
            tag="crossing_corridor",
        )

        lane_edge_roi = SceneEditROI(
            x_min=conflict_x - 1.5,
            y_min=lane_y - self._lane_half_width_m - 0.8,
            x_max=conflict_x + 1.5,
            y_max=lane_y + self._lane_half_width_m + 0.8,
            tag="lane_edge_conflict_anchor",
        )

        roadside_anchor_roi = SceneEditROI(
            x_min=x - 1.2,
            y_min=y - 0.8,
            x_max=x + 1.2,
            y_max=y + 0.8,
            tag="roadside_spawn_anchor",
        )

        return [ped_roi, corridor_roi, lane_edge_roi, roadside_anchor_roi]

    def _allocate_slot(self, elem: SledgeVectorElement) -> int:
        valid = np.asarray(elem.mask).astype(bool)
        invalid = np.where(~valid)[0]
        if len(invalid) > 0:
            idx = int(invalid[0])
            elem.mask[idx] = True
            return idx

        distances = np.linalg.norm(np.asarray(elem.states)[:, :2], axis=-1)
        idx = int(np.argmax(distances))
        elem.mask[idx] = True
        return idx

    def _set_agent_state(
        self,
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

    @staticmethod
    def _valid_agent_states(elem: SledgeVectorElement) -> np.ndarray:
        valid = np.asarray(elem.mask).astype(bool)
        states = np.asarray(elem.states)
        return states[valid] if np.any(valid) else np.zeros((0, states.shape[-1]), dtype=states.dtype)

    @staticmethod
    def _deepcopy_scene(scene: SledgeVectorRaw) -> SledgeVectorRaw:
        return copy.deepcopy(scene)
