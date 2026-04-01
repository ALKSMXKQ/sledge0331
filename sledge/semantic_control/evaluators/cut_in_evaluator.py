from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List

import numpy as np

from sledge.autoencoder.preprocessing.features.sledge_vector_feature import AgentIndex, SledgeVector

LABEL_THRESH = 0.3

# V3 evaluator: less obsessed with exact adjacent-lane center,
# more focused on whether the scene visibly behaves like a front cut-in proxy.
_SEVERITY_TARGETS = {
    "mild": {
        "t_center_peak": 2.2,
        "t_center_half_width": 0.8,
        "cross_x_peak": 11.0,
        "cross_x_half_width": 4.0,
        "start_x_low": 0.0,
        "start_x_high": 9.0,
    },
    "moderate": {
        "t_center_peak": 1.5,
        "t_center_half_width": 0.5,
        "cross_x_peak": 8.0,
        "cross_x_half_width": 3.0,
        "start_x_low": -2.0,
        "start_x_high": 7.0,
    },
    "aggressive": {
        "t_center_peak": 1.0,
        "t_center_half_width": 0.35,
        "cross_x_peak": 5.0,
        "cross_x_half_width": 2.0,
        "start_x_low": -4.0,
        "start_x_high": 5.0,
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


class CutInAlignmentEvaluator:
    def __init__(
        self,
        lane_half_width_m: float = 1.8,
        prediction_horizon_s: float = 4.0,
        prediction_dt_s: float = 0.1,
    ) -> None:
        self.lane_half_width_m = lane_half_width_m
        self.prediction_horizon_s = prediction_horizon_s
        self.prediction_dt_s = prediction_dt_s
        self._ego_lane_y = 0.0

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
                    "front_insert_score": 0.0,
                    "cut_in_immediacy_score": 0.0,
                    "pre_merge_proximity_score": 0.0,
                },
                notes=["no valid vehicle detected"],
                accepted=False,
            )

        severity = getattr(prompt_spec, "severity_level", "moderate") if prompt_spec is not None else "moderate"
        target = _SEVERITY_TARGETS.get(severity, _SEVERITY_TARGETS["moderate"])
        lane_y = self._ego_lane_y

        candidate_metrics: List[Dict[str, Any]] = []

        for veh in vehicles:
            metrics = self._score_cut_in_vehicle(veh, lane_y, target)
            if metrics["vehicle_presence_score"] > 0:
                candidate_metrics.append(metrics)

        if len(candidate_metrics) == 0:
            return PromptAlignmentResult(
                total=0.0,
                details={
                    "vehicle_presence_score": 0.0,
                    "adjacent_origin_score": 0.0,
                    "merge_direction_score": 0.0,
                    "front_insert_score": 0.0,
                    "cut_in_immediacy_score": 0.0,
                    "pre_merge_proximity_score": 0.0,
                },
                notes=["no plausible cut-in candidate vehicle found"],
                accepted=False,
            )

        best = max(
            candidate_metrics,
            key=lambda m: (
                0.10 * m["vehicle_presence_score"]
                + 0.22 * m["adjacent_origin_score"]
                + 0.20 * m["merge_direction_score"]
                + 0.25 * m["front_insert_score"]
                + 0.13 * m["cut_in_immediacy_score"]
                + 0.10 * m["pre_merge_proximity_score"]
            ),
        )

        raw_total = (
            0.10 * best["vehicle_presence_score"]
            + 0.22 * best["adjacent_origin_score"]
            + 0.20 * best["merge_direction_score"]
            + 0.25 * best["front_insert_score"]
            + 0.13 * best["cut_in_immediacy_score"]
            + 0.10 * best["pre_merge_proximity_score"]
        )

        # Softer gates than before
        origin_gate = 1.0 if best["adjacent_origin_score"] >= 0.35 else 0.35
        direction_gate = 1.0 if best["merge_direction_score"] >= 0.35 else 0.45
        insert_gate = 1.0 if best["front_insert_score"] >= 0.40 else 0.30

        total = float(np.clip(raw_total * origin_gate * direction_gate * insert_gate, 0.0, 1.0))

        notes.extend(best["notes"])
        notes.append(f"cut-in semantics enabled: practical front-insert proxy ({severity})")

        details = {
            "vehicle_presence_score": best["vehicle_presence_score"],
            "adjacent_origin_score": best["adjacent_origin_score"],
            "merge_direction_score": best["merge_direction_score"],
            "front_insert_score": best["front_insert_score"],
            "cut_in_immediacy_score": best["cut_in_immediacy_score"],
            "pre_merge_proximity_score": best["pre_merge_proximity_score"],
        }

        accepted = total >= 0.62
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
            valid = bool(mask) if isinstance(mask, (bool, np.bool_)) else float(mask) >= LABEL_THRESH
            if valid:
                vehicles.append(np.asarray(state, dtype=np.float32))
        return vehicles

    def _score_cut_in_vehicle(self, veh: np.ndarray, lane_y: float, target: Dict[str, float]) -> Dict[str, Any]:
        x0 = float(veh[AgentIndex.X])
        y0 = float(veh[AgentIndex.Y])
        heading = float(veh[AgentIndex.HEADING])
        speed = float(max(0.0, veh[AgentIndex.VELOCITY]))

        # Hard plausibility filter first
        if abs(y0 - lane_y) < 2.0:
            return self._zero_metrics("vehicle starts too close to ego lane center")
        if x0 < -10.0 or x0 > 18.0:
            return self._zero_metrics("vehicle starts outside practical cut-in longitudinal range")

        vx = speed * math.cos(heading)
        vy = speed * math.sin(heading)

        toward_ego_lane = (y0 > lane_y and vy < 0.0) or (y0 < lane_y and vy > 0.0)
        if not toward_ego_lane:
            return self._zero_metrics("vehicle is not moving toward ego lane")

        times = np.arange(0.0, self.prediction_horizon_s + 1e-6, self.prediction_dt_s, dtype=np.float32)
        future_x = x0 + vx * times
        future_y = y0 + vy * times

        vehicle_presence_score = 1.0

        adjacent_origin_score = self._band_score(abs(y0 - lane_y), low=2.2, high=5.2)

        lateral_ratio = abs(vy) / max(1e-3, abs(vx) + abs(vy))
        merge_direction_score = float(
            np.clip(0.6 * 1.0 + 0.4 * np.clip(lateral_ratio / 0.18, 0.0, 1.0), 0.0, 1.0)
        )

        # More visible proxy: use crossing near ego-lane center band, not merely touching boundary
        center_band = np.abs(future_y - lane_y) <= 0.8

        if np.any(center_band):
            idx_cross = int(np.argmax(center_band))
            t_cross = float(times[idx_cross])
            x_cross = float(future_x[idx_cross])

            front_insert_score = self._triangular_score(
                x_cross,
                peak=target["cross_x_peak"],
                half_width=target["cross_x_half_width"],
            )
            if x_cross < 1.5:
                front_insert_score *= 0.1

            cut_in_immediacy_score = self._triangular_score(
                t_cross,
                peak=target["t_center_peak"],
                half_width=target["t_center_half_width"],
            )
        else:
            t_cross = float("inf")
            x_cross = float("-inf")
            front_insert_score = 0.0
            min_center_dist = float(np.min(np.abs(future_y - lane_y)))
            cut_in_immediacy_score = float(np.clip(1.0 - min_center_dist / 3.0, 0.0, 0.30))

        pre_merge_proximity_score = self._band_score(
            x0,
            low=target["start_x_low"],
            high=target["start_x_high"],
        )

        notes: List[str] = []
        if adjacent_origin_score < 0.35:
            notes.append("vehicle does not clearly start from an adjacent lane")
        if merge_direction_score < 0.35:
            notes.append("vehicle motion is not clearly directed into ego lane")
        if front_insert_score < 0.40:
            notes.append("vehicle does not clearly insert into ego lane ahead of ego")
        if cut_in_immediacy_score < 0.30:
            notes.append("cut-in timing does not match the requested severity tier")
        if pre_merge_proximity_score < 0.30:
            notes.append("vehicle starts too far away longitudinally to look like a real cut-in")

        notes.append(
            f"debug: start=(x={x0:.2f}, y={y0:.2f}), vx={vx:.2f}, vy={vy:.2f}, "
            f"t_cross={t_cross if np.isfinite(t_cross) else -1:.2f}, "
            f"x_cross={x_cross if np.isfinite(x_cross) else -999:.2f}"
        )

        return {
            "vehicle_presence_score": float(vehicle_presence_score),
            "adjacent_origin_score": float(np.clip(adjacent_origin_score, 0.0, 1.0)),
            "merge_direction_score": float(np.clip(merge_direction_score, 0.0, 1.0)),
            "front_insert_score": float(np.clip(front_insert_score, 0.0, 1.0)),
            "cut_in_immediacy_score": float(np.clip(cut_in_immediacy_score, 0.0, 1.0)),
            "pre_merge_proximity_score": float(np.clip(pre_merge_proximity_score, 0.0, 1.0)),
            "notes": notes,
        }

    @staticmethod
    def _zero_metrics(reason: str) -> Dict[str, Any]:
        return {
            "vehicle_presence_score": 0.0,
            "adjacent_origin_score": 0.0,
            "merge_direction_score": 0.0,
            "front_insert_score": 0.0,
            "cut_in_immediacy_score": 0.0,
            "pre_merge_proximity_score": 0.0,
            "notes": [reason],
        }

    @staticmethod
    def _band_score(value: float, low: float, high: float) -> float:
        if low <= value <= high:
            return 1.0
        if value < low:
            return float(np.clip(1.0 - (low - value) / max(1e-6, low), 0.0, 1.0))
        return float(np.clip(1.0 - (value - high) / max(1e-6, high), 0.0, 1.0))

    @staticmethod
    def _triangular_score(value: float, peak: float, half_width: float) -> float:
        if half_width <= 0:
            return 0.0
        return float(np.clip(1.0 - abs(value - peak) / half_width, 0.0, 1.0))