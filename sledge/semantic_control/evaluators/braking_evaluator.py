from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List

import numpy as np

from sledge.autoencoder.preprocessing.features.sledge_vector_feature import AgentIndex, SledgeVector

LABEL_THRESH = 0.3

_SEVERITY_TARGETS = {
    "mild": {"distance_peak": 16.0, "distance_half_width": 6.0, "speed_ratio_peak": 0.60},
    "moderate": {"distance_peak": 10.0, "distance_half_width": 4.0, "speed_ratio_peak": 0.40},
    "aggressive": {"distance_peak": 6.0, "distance_half_width": 3.0, "speed_ratio_peak": 0.20},
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


class HardBrakeAlignmentEvaluator:
    def __init__(self, lane_half_width_m: float = 1.8) -> None:
        self.lane_half_width_m = lane_half_width_m

    def evaluate(self, sledge_vector: SledgeVector, prompt_spec: Any = None) -> PromptAlignmentResult:
        vehicles = self._collect_valid_vehicles(sledge_vector)
        notes: List[str] = []
        if len(vehicles) == 0:
            return PromptAlignmentResult(
                total=0.0,
                details={
                    "lead_vehicle_presence_score": 0.0,
                    "same_lane_score": 0.0,
                    "short_headway_score": 0.0,
                    "low_speed_score": 0.0,
                    "braking_immediacy_score": 0.0,
                },
                notes=["no valid lead vehicle detected"],
                accepted=False,
            )

        severity = getattr(prompt_spec, "severity_level", "moderate") if prompt_spec is not None else "moderate"
        target = _SEVERITY_TARGETS.get(severity, _SEVERITY_TARGETS["moderate"])
        lane_y = self._estimate_lane_center_y(vehicles)
        ego_speed = self._extract_ego_speed(sledge_vector)

        best_score = -1.0
        best = None
        for veh in vehicles:
            metrics = self._score_lead_vehicle(veh, lane_y, ego_speed, target)
            composite = (
                0.15 * metrics["lead_vehicle_presence_score"]
                + 0.25 * metrics["same_lane_score"]
                + 0.25 * metrics["short_headway_score"]
                + 0.20 * metrics["low_speed_score"]
                + 0.15 * metrics["braking_immediacy_score"]
            )
            if composite > best_score:
                best_score = composite
                best = metrics

        assert best is not None
        lane_gate = 1.0 if best["same_lane_score"] >= 0.45 else 0.2
        headway_gate = 1.0 if best["short_headway_score"] >= 0.45 else 0.2
        total = float(np.clip(best_score * lane_gate * headway_gate, 0.0, 1.0))
        notes.extend(best["notes"])
        notes.append(f"hard-brake semantics enabled: severity-aware gating ({severity})")

        details = {
            "lead_vehicle_presence_score": best["lead_vehicle_presence_score"],
            "same_lane_score": best["same_lane_score"],
            "short_headway_score": best["short_headway_score"],
            "low_speed_score": best["low_speed_score"],
            "braking_immediacy_score": best["braking_immediacy_score"],
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

    def _score_lead_vehicle(self, veh: np.ndarray, lane_y: float, ego_speed: float, target: Dict[str, float]) -> Dict[str, Any]:
        x = float(veh[AgentIndex.X])
        y = float(veh[AgentIndex.Y])
        speed = float(max(0.0, veh[AgentIndex.VELOCITY]))

        lead_vehicle_presence_score = 1.0
        same_lane_score = float(np.clip(1.0 - abs(y - lane_y) / (self.lane_half_width_m + 0.5), 0.0, 1.0))
        short_headway_score = self._triangular_score(x, peak=target["distance_peak"], half_width=target["distance_half_width"])

        speed_ratio = speed / max(ego_speed, 1e-3)
        low_speed_score = float(np.clip(1.0 - abs(speed_ratio - target["speed_ratio_peak"]) / 0.4, 0.0, 1.0))

        relative_closure = max(0.0, ego_speed - speed)
        ttc = x / max(relative_closure, 1e-3) if x > 0.0 else 0.0
        braking_immediacy_score = self._triangular_score(ttc, peak=2.0 if target["speed_ratio_peak"] < 0.3 else 3.0, half_width=1.0)

        notes: List[str] = []
        if same_lane_score < 0.45:
            notes.append("lead vehicle is not clearly in ego lane")
        if short_headway_score < 0.45:
            notes.append("lead vehicle is not close enough ahead")
        if low_speed_score < 0.45:
            notes.append("lead vehicle is not sufficiently slower than ego")
        if braking_immediacy_score < 0.35:
            notes.append("hard-brake urgency does not match the requested tier")

        return {
            "lead_vehicle_presence_score": lead_vehicle_presence_score,
            "same_lane_score": same_lane_score,
            "short_headway_score": short_headway_score,
            "low_speed_score": low_speed_score,
            "braking_immediacy_score": braking_immediacy_score,
            "notes": notes,
        }

    @staticmethod
    def _triangular_score(value: float, peak: float, half_width: float) -> float:
        if half_width <= 0:
            return 0.0
        return float(np.clip(1.0 - abs(value - peak) / half_width, 0.0, 1.0))
