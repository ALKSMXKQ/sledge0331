from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np

from sledge.autoencoder.preprocessing.features.sledge_vector_feature import AgentIndex, SledgeVector

LABEL_THRESH = 0.3

_SEVERITY_TARGETS = {
    "mild": {
        "distance_peak": 15.5,
        "distance_half_width": 6.0,
        "speed_ratio_peak": 0.62,
        "ttc_peak": 3.3,
        "ttc_half_width": 1.4,
    },
    "moderate": {
        "distance_peak": 10.5,
        "distance_half_width": 4.5,
        "speed_ratio_peak": 0.42,
        "ttc_peak": 2.4,
        "ttc_half_width": 1.0,
    },
    "aggressive": {
        "distance_peak": 6.8,
        "distance_half_width": 3.2,
        "speed_ratio_peak": 0.22,
        "ttc_peak": 1.5,
        "ttc_half_width": 0.7,
    },
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
    """
    More stable hard-brake evaluator.

    Main changes:
      1) Lead selection is ego-centric, not global-median-lane-centric.
      2) TTC target is explicitly aligned with the editor / severity.
      3) Outputs include candidate diagnostics for easier manifest analysis.
    """

    def __init__(self, lane_half_width_m: float = 1.8) -> None:
        self.lane_half_width_m = lane_half_width_m
        self.same_lane_thresh_m = 1.35

    def evaluate(self, sledge_vector: SledgeVector, prompt_spec: Any = None) -> PromptAlignmentResult:
        vehicles = self._collect_valid_vehicles(sledge_vector)
        if len(vehicles) == 0:
            return PromptAlignmentResult(
                total=0.0,
                details={
                    "lead_vehicle_presence_score": 0.0,
                    "same_lane_score": 0.0,
                    "short_headway_score": 0.0,
                    "low_speed_score": 0.0,
                    "braking_immediacy_score": 0.0,
                    "candidate_x_m": -1.0,
                    "candidate_y_m": 99.0,
                },
                notes=["no valid lead vehicle detected"],
                accepted=False,
            )

        severity = getattr(prompt_spec, "severity_level", "moderate") if prompt_spec is not None else "moderate"
        target = dict(_SEVERITY_TARGETS.get(severity, _SEVERITY_TARGETS["moderate"]))

        # Prefer prompt-provided targets when present.
        if prompt_spec is not None:
            dist_min = float(getattr(prompt_spec, "lead_distance_min_m", target["distance_peak"] - target["distance_half_width"]))
            dist_max = float(getattr(prompt_spec, "lead_distance_max_m", target["distance_peak"] + target["distance_half_width"]))
            target["distance_peak"] = 0.5 * (dist_min + dist_max)
            target["distance_half_width"] = max(1.0, 0.5 * abs(dist_max - dist_min))
            ttc_min = float(getattr(prompt_spec, "ttc_min_s", target["ttc_peak"] - 0.5))
            ttc_max = float(getattr(prompt_spec, "ttc_max_s", target["ttc_peak"] + 0.5))
            target["ttc_peak"] = 0.5 * (ttc_min + ttc_max)
            target["ttc_half_width"] = max(0.5, 0.5 * abs(ttc_max - ttc_min))

        ego_speed = self._extract_ego_speed(sledge_vector)
        lane_y = self._estimate_ego_lane_y(sledge_vector)

        candidates = self._find_forward_candidates(vehicles, lane_y)
        if not candidates:
            return PromptAlignmentResult(
                total=0.0,
                details={
                    "lead_vehicle_presence_score": 0.0,
                    "same_lane_score": 0.0,
                    "short_headway_score": 0.0,
                    "low_speed_score": 0.0,
                    "braking_immediacy_score": 0.0,
                    "candidate_x_m": -1.0,
                    "candidate_y_m": 99.0,
                },
                notes=["no forward candidate ahead of ego in the target lane band"],
                accepted=False,
            )

        best = None
        best_score = -1.0
        for veh in candidates:
            metrics = self._score_lead_vehicle(veh, lane_y, ego_speed, target)
            composite = (
                0.15 * metrics["lead_vehicle_presence_score"]
                + 0.25 * metrics["same_lane_score"]
                + 0.25 * metrics["short_headway_score"]
                + 0.15 * metrics["low_speed_score"]
                + 0.20 * metrics["braking_immediacy_score"]
            )
            if composite > best_score:
                best_score = composite
                best = metrics

        assert best is not None

        lane_gate = 1.0 if best["same_lane_score"] >= 0.45 else 0.2
        headway_gate = 1.0 if best["short_headway_score"] >= 0.45 else 0.2
        total = float(np.clip(best_score * lane_gate * headway_gate, 0.0, 1.0))
        notes = list(best["notes"])
        notes.append(f"hard-brake semantics enabled: ego-centric lead selection ({severity})")

        details = {
            "lead_vehicle_presence_score": best["lead_vehicle_presence_score"],
            "same_lane_score": best["same_lane_score"],
            "short_headway_score": best["short_headway_score"],
            "low_speed_score": best["low_speed_score"],
            "braking_immediacy_score": best["braking_immediacy_score"],
            "candidate_x_m": best["candidate_x_m"],
            "candidate_y_m": best["candidate_y_m"],
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

    def _estimate_ego_lane_y(self, sledge_vector: SledgeVector) -> float:
        vehicles = self._collect_valid_vehicles(sledge_vector)
        if len(vehicles) == 0:
            return 0.0
        arr = np.asarray(vehicles)
        same_band = arr[np.abs(arr[:, AgentIndex.Y]) < 2.2]
        if len(same_band) == 0:
            return 0.0
        ahead = same_band[(same_band[:, AgentIndex.X] > -5.0) & (same_band[:, AgentIndex.X] < 35.0)]
        if len(ahead) == 0:
            return 0.0
        return float(np.clip(np.median(ahead[:, AgentIndex.Y]), -0.8, 0.8))

    def _find_forward_candidates(self, vehicles: List[np.ndarray], lane_y: float) -> List[np.ndarray]:
        arr = np.asarray(vehicles, dtype=np.float32)
        if arr.size == 0:
            return []
        ahead = arr[arr[:, AgentIndex.X] > 1.5]
        if len(ahead) == 0:
            return []
        same_band = ahead[np.abs(ahead[:, AgentIndex.Y] - lane_y) < (self.lane_half_width_m + 1.0)]
        if len(same_band) == 0:
            return list(ahead)
        return list(same_band)

    def _extract_ego_speed(self, sledge_vector: SledgeVector) -> float:
        ego_states = np.asarray(sledge_vector.ego.states).reshape(-1)
        if ego_states.size == 0:
            return 5.0
        return float(np.clip(abs(ego_states[0]), 2.0, 15.0))

    def _score_lead_vehicle(
        self,
        veh: np.ndarray,
        lane_y: float,
        ego_speed: float,
        target: Dict[str, float],
    ) -> Dict[str, Any]:
        x = float(veh[AgentIndex.X])
        y = float(veh[AgentIndex.Y])
        speed = float(max(0.0, veh[AgentIndex.VELOCITY]))

        lead_vehicle_presence_score = 1.0
        same_lane_score = float(np.clip(1.0 - abs(y - lane_y) / self.same_lane_thresh_m, 0.0, 1.0))
        short_headway_score = self._triangular_score(
            x,
            peak=target["distance_peak"],
            half_width=target["distance_half_width"],
        )

        speed_ratio = speed / max(ego_speed, 1e-3)
        low_speed_score = float(np.clip(1.0 - abs(speed_ratio - target["speed_ratio_peak"]) / 0.35, 0.0, 1.0))

        relative_closure = max(ego_speed - speed, 0.0)
        if x <= 0.0:
            ttc = 0.0
        elif relative_closure <= 1e-3:
            ttc = 999.0
        else:
            ttc = x / relative_closure

        braking_immediacy_score = self._triangular_score(
            ttc,
            peak=target["ttc_peak"],
            half_width=target["ttc_half_width"],
        )

        notes: List[str] = []
        if same_lane_score < 0.45:
            notes.append("lead vehicle is not clearly in ego lane")
        if short_headway_score < 0.45:
            notes.append("lead vehicle is not close enough ahead")
        if low_speed_score < 0.45:
            notes.append("lead vehicle is not sufficiently slower than ego")
        if braking_immediacy_score < 0.35:
            notes.append("hard-brake TTC / urgency does not match the requested tier")

        return {
            "lead_vehicle_presence_score": lead_vehicle_presence_score,
            "same_lane_score": same_lane_score,
            "short_headway_score": short_headway_score,
            "low_speed_score": low_speed_score,
            "braking_immediacy_score": braking_immediacy_score,
            "candidate_x_m": x,
            "candidate_y_m": y,
            "notes": notes,
        }

    @staticmethod
    def _triangular_score(value: float, peak: float, half_width: float) -> float:
        if half_width <= 0:
            return 0.0
        return float(np.clip(1.0 - abs(value - peak) / half_width, 0.0, 1.0))
