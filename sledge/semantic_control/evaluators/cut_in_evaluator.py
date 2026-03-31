from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List

import numpy as np

from sledge.autoencoder.preprocessing.features.sledge_vector_feature import AgentIndex, SledgeVector

LABEL_THRESH = 0.3

_SEVERITY_TARGETS = {
    "mild": {"gap_peak": 12.0, "gap_half_width": 4.0},
    "moderate": {"gap_peak": 8.0, "gap_half_width": 3.0},
    "aggressive": {"gap_peak": 5.0, "gap_half_width": 2.0},
}


@dataclass
class PromptAlignmentResult:
    total: float
    details: Dict[str, float]
    notes: List[str] = field(default_factory=list)
    accepted: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total": float(self.total),
            "details": {k: float(v) for k, v in self.details.items()},
            "notes": list(self.notes),
            "accepted": bool(self.accepted),
        }


class CutInAlignmentEvaluator:
    def __init__(self, lane_half_width_m: float = 1.8, adjacent_lane_min_offset_m: float = 2.2, prediction_horizon_s: float = 4.0, prediction_dt_s: float = 0.1) -> None:
        self.lane_half_width_m = lane_half_width_m
        self.adjacent_lane_min_offset_m = adjacent_lane_min_offset_m
        self.prediction_horizon_s = prediction_horizon_s
        self.prediction_dt_s = prediction_dt_s

    def evaluate(self, sledge_vector: SledgeVector, prompt_spec: Any = None) -> PromptAlignmentResult:
        vehicles = self._collect_valid_vehicles(sledge_vector)
        notes: List[str] = []
        if len(vehicles) == 0:
            return PromptAlignmentResult(
                total=0.0,
                details={
                    "vehicle_presence_score": 0.0,
                    "adjacent_origin_score": 0.0,
                    "merge_direction_score": 0.0,
                    "ego_lane_merge_conflict_score": 0.0,
                    "cut_in_immediacy_score": 0.0,
                },
                notes=["no valid vehicle detected"],
                accepted=False,
            )

        severity = getattr(prompt_spec, "severity_level", "moderate") if prompt_spec is not None else "moderate"
        target = _SEVERITY_TARGETS.get(severity, _SEVERITY_TARGETS["moderate"])
        lane_y = self._estimate_lane_center_y(vehicles)
        ego_speed = self._extract_ego_speed(sledge_vector)

        best_score = -1.0
        best = None
        for veh in vehicles:
            metrics = self._score_cut_in_vehicle(veh, lane_y, ego_speed, target)
            composite = (
                0.15 * metrics["vehicle_presence_score"]
                + 0.20 * metrics["adjacent_origin_score"]
                + 0.20 * metrics["merge_direction_score"]
                + 0.30 * metrics["ego_lane_merge_conflict_score"]
                + 0.15 * metrics["cut_in_immediacy_score"]
            )
            if composite > best_score:
                best_score = composite
                best = metrics

        assert best is not None
        origin_gate = 1.0 if best["adjacent_origin_score"] >= 0.45 else 0.2
        merge_gate = 1.0 if best["ego_lane_merge_conflict_score"] >= 0.45 else 0.2
        total = float(np.clip(best_score * origin_gate * merge_gate, 0.0, 1.0))
        notes.extend(best["notes"])
        notes.append(f"cut-in semantics enabled: severity-aware gating ({severity})")

        details = {
            "vehicle_presence_score": best["vehicle_presence_score"],
            "adjacent_origin_score": best["adjacent_origin_score"],
            "merge_direction_score": best["merge_direction_score"],
            "ego_lane_merge_conflict_score": best["ego_lane_merge_conflict_score"],
            "cut_in_immediacy_score": best["cut_in_immediacy_score"],
        }
        accepted = total >= 0.7
        return PromptAlignmentResult(total=total, details=details, notes=notes, accepted=accepted)

    def _collect_valid_vehicles(self, sledge_vector: SledgeVector) -> List[np.ndarray]:
        states = np.asarray(sledge_vector.vehicles.states)
        masks = np.asarray(sledge_vector.vehicles.mask)
        if states.size == 0:
            return []
        if states.ndim == 1:
            states = states[None, :]
        if masks.ndim == 0:
            masks = np.asarray([masks])

        vehicles: List[np.ndarray] = []
        for state, mask in zip(states, masks):
            valid = (bool(mask) if isinstance(mask, (bool, np.bool_)) else float(mask) >= LABEL_THRESH)
            if valid:
                vehicles.append(np.asarray(state, dtype=np.float32))
        return vehicles

    def _estimate_lane_center_y(self, vehicles: List[np.ndarray]) -> float:
        arr = np.asarray(vehicles)
        usable = arr[np.abs(arr[:, AgentIndex.Y]) < 4.0]
        if len(usable) == 0:
            usable = arr
        return float(np.median(usable[:, AgentIndex.Y])) if len(usable) > 0 else 0.0

    def _extract_ego_speed(self, sledge_vector: SledgeVector) -> float:
        ego_states = np.asarray(sledge_vector.ego.states).reshape(-1)
        if ego_states.size == 0:
            return 5.0
        return float(abs(ego_states[0]))

    def _score_cut_in_vehicle(self, veh: np.ndarray, lane_y: float, ego_speed: float, target: Dict[str, float]) -> Dict[str, Any]:
        x = float(veh[AgentIndex.X])
        y = float(veh[AgentIndex.Y])
        heading = float(veh[AgentIndex.HEADING])
        speed = float(max(0.0, veh[AgentIndex.VELOCITY]))

        vx = speed * math.cos(heading)
        vy = speed * math.sin(heading)
        times = np.arange(0.0, self.prediction_horizon_s + 1e-6, self.prediction_dt_s, dtype=np.float32)
        future_x = x + vx * times
        future_y = y + vy * times

        vehicle_presence_score = 1.0
        adjacent_origin_score = float(np.clip((abs(y - lane_y) - self.lane_half_width_m) / max(1e-3, self.adjacent_lane_min_offset_m), 0.0, 1.0))
        toward_ego_lane = 1.0 if (y > lane_y and vy < 0.0) or (y < lane_y and vy > 0.0) else 0.0
        lateral_ratio = abs(vy) / max(1e-3, abs(vx) + abs(vy))
        merge_direction_score = float(np.clip(0.7 * lateral_ratio + 0.3 * toward_ego_lane, 0.0, 1.0))

        enters_ego_lane = np.abs(future_y - lane_y) <= self.lane_half_width_m
        ahead_of_ego = future_x >= 2.0
        merge_conflict = enters_ego_lane & ahead_of_ego
        ego_lane_merge_conflict_score = 1.0 if np.any(merge_conflict) else 0.0

        if np.any(merge_conflict):
            merge_x = float(future_x[np.argmax(merge_conflict)])
            cut_in_immediacy_score = self._triangular_score(merge_x, peak=target["gap_peak"], half_width=target["gap_half_width"])
        else:
            min_lane_dist = float(np.min(np.abs(future_y - lane_y)))
            cut_in_immediacy_score = float(np.clip(1.0 - min_lane_dist / 3.0, 0.0, 0.4))

        notes: List[str] = []
        if adjacent_origin_score < 0.45:
            notes.append("vehicle is not clearly starting from adjacent lane")
        if merge_direction_score < 0.45:
            notes.append("vehicle motion is not clearly merging into ego lane")
        if ego_lane_merge_conflict_score < 0.45:
            notes.append("predicted vehicle path does not enter ego lane ahead")
        if cut_in_immediacy_score < 0.35:
            notes.append("cut-in gap / urgency does not match the requested tier")

        return {
            "vehicle_presence_score": vehicle_presence_score,
            "adjacent_origin_score": adjacent_origin_score,
            "merge_direction_score": merge_direction_score,
            "ego_lane_merge_conflict_score": ego_lane_merge_conflict_score,
            "cut_in_immediacy_score": cut_in_immediacy_score,
            "notes": notes,
        }

    @staticmethod
    def _triangular_score(value: float, peak: float, half_width: float) -> float:
        if half_width <= 0:
            return 0.0
        return float(np.clip(1.0 - abs(value - peak) / half_width, 0.0, 1.0))
