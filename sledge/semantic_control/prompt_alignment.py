from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List

import numpy as np

from sledge.autoencoder.preprocessing.features.sledge_vector_feature import AgentIndex, SledgeVector

LABEL_THRESH = 0.3


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


_SEVERITY_TARGETS = {
    "mild": {"ttc_peak": 3.5, "ttc_half_width": 1.0},
    "moderate": {"ttc_peak": 2.5, "ttc_half_width": 0.8},
    "aggressive": {"ttc_peak": 1.6, "ttc_half_width": 0.6},
}


class PromptAlignmentEvaluator:
    """
    Evaluator aligned with:
        "突发的行人横穿马路"

    This version is severity-aware:
        - mild
        - moderate
        - aggressive
    """

    def __init__(
        self,
        lane_half_width_m: float = 1.8,
        roadside_min_y_m: float = 2.0,
        roadside_max_y_m: float = 8.0,
        forward_min_x_m: float = 2.0,
        forward_max_x_m: float = 24.0,
        conflict_x_min_m: float = 4.0,
        conflict_x_max_m: float = 20.0,
        prediction_horizon_s: float = 4.5,
        prediction_dt_s: float = 0.1,
    ) -> None:
        self.lane_half_width_m = lane_half_width_m
        self.roadside_min_y_m = roadside_min_y_m
        self.roadside_max_y_m = roadside_max_y_m
        self.forward_min_x_m = forward_min_x_m
        self.forward_max_x_m = forward_max_x_m
        self.conflict_x_min_m = conflict_x_min_m
        self.conflict_x_max_m = conflict_x_max_m
        self.prediction_horizon_s = prediction_horizon_s
        self.prediction_dt_s = prediction_dt_s

    def evaluate(self, sledge_vector: SledgeVector, prompt_spec: Any = None) -> PromptAlignmentResult:
        pedestrians = self._collect_valid_pedestrians(sledge_vector)
        notes: List[str] = []

        if len(pedestrians) == 0:
            return PromptAlignmentResult(
                total=0.0,
                details={
                    "pedestrian_presence_score": 0.0,
                    "roadside_emergence_score": 0.0,
                    "crossing_direction_score": 0.0,
                    "ego_lane_conflict_score": 0.0,
                    "immediacy_score": 0.0,
                },
                notes=["no valid pedestrian detected"],
                accepted=False,
            )

        severity = getattr(prompt_spec, "severity_level", "moderate") if prompt_spec is not None else "moderate"
        target = _SEVERITY_TARGETS.get(severity, _SEVERITY_TARGETS["moderate"])
        ego_speed = self._extract_ego_speed(sledge_vector)
        best = None
        best_score = -1.0

        for ped in pedestrians:
            ped_metrics = self._score_pedestrian_crossing(ped, ego_speed, target)
            composite = (
                0.15 * ped_metrics["pedestrian_presence_score"]
                + 0.20 * ped_metrics["roadside_emergence_score"]
                + 0.20 * ped_metrics["crossing_direction_score"]
                + 0.30 * ped_metrics["ego_lane_conflict_score"]
                + 0.15 * ped_metrics["immediacy_score"]
            )
            if composite > best_score:
                best_score = composite
                best = ped_metrics

        assert best is not None

        crossing_prompt = self._is_crossing_prompt(prompt_spec)
        if crossing_prompt:
            roadside_gate = 1.0 if best["roadside_emergence_score"] >= 0.45 else 0.2
            conflict_gate = 1.0 if best["ego_lane_conflict_score"] >= 0.45 else 0.2
            total = best_score * roadside_gate * conflict_gate
        else:
            total = best_score

        notes.extend(best.get("notes", []))
        if crossing_prompt:
            notes.append(f"crossing prompt semantics enabled: severity-aware gating ({severity})")

        details = {
            "pedestrian_presence_score": best["pedestrian_presence_score"],
            "roadside_emergence_score": best["roadside_emergence_score"],
            "crossing_direction_score": best["crossing_direction_score"],
            "ego_lane_conflict_score": best["ego_lane_conflict_score"],
            "immediacy_score": best["immediacy_score"],
        }
        total = float(np.clip(total, 0.0, 1.0))
        accepted = total >= 0.7
        return PromptAlignmentResult(total=total, details=details, notes=notes, accepted=accepted)

    def _collect_valid_pedestrians(self, sledge_vector: SledgeVector) -> List[np.ndarray]:
        states = np.asarray(sledge_vector.pedestrians.states)
        masks = np.asarray(sledge_vector.pedestrians.mask)

        if states.size == 0:
            return []
        if states.ndim == 1:
            states = states[None, :]
        if masks.ndim == 0:
            masks = np.asarray([masks])

        pedestrians: List[np.ndarray] = []
        for state, mask in zip(states, masks):
            valid = (bool(mask) if isinstance(mask, (bool, np.bool_)) else float(mask) >= LABEL_THRESH)
            if valid:
                pedestrians.append(np.asarray(state, dtype=np.float32))
        return pedestrians

    def _extract_ego_speed(self, sledge_vector: SledgeVector) -> float:
        ego_states = np.asarray(sledge_vector.ego.states)
        if ego_states.size == 0:
            return 5.0
        flat = ego_states.reshape(-1)
        return float(abs(flat[0])) if flat.size > 0 else 5.0

    def _score_pedestrian_crossing(self, ped_state: np.ndarray, ego_speed: float, target: Dict[str, float]) -> Dict[str, Any]:
        x = float(ped_state[AgentIndex.X])
        y = float(ped_state[AgentIndex.Y])
        heading = float(ped_state[AgentIndex.HEADING])
        speed = float(max(0.0, ped_state[AgentIndex.VELOCITY]))

        vx = speed * math.cos(heading)
        vy = speed * math.sin(heading)
        times = np.arange(0.0, self.prediction_horizon_s + 1e-6, self.prediction_dt_s, dtype=np.float32)
        future_x = x + vx * times
        future_y = y + vy * times

        pedestrian_presence_score = 1.0
        abs_y = abs(y)
        roadside_band_score = self._band_score(abs_y, self.roadside_min_y_m, self.roadside_max_y_m)
        forward_start_score = self._band_score(x, self.forward_min_x_m, self.forward_max_x_m)
        roadside_emergence_score = 0.6 * roadside_band_score + 0.4 * forward_start_score

        total_speed = max(1e-3, abs(vx) + abs(vy))
        lateral_ratio = abs(vy) / total_speed
        toward_center = 1.0 if (y > 0.0 and vy < 0.0) or (y < 0.0 and vy > 0.0) else 0.0
        crossing_direction_score = np.clip(0.7 * lateral_ratio + 0.3 * toward_center, 0.0, 1.0)

        in_lane = np.abs(future_y) <= self.lane_half_width_m
        in_conflict_x = (future_x >= self.conflict_x_min_m) & (future_x <= self.conflict_x_max_m)
        enters_conflict = in_lane & in_conflict_x
        ego_lane_conflict_score = 1.0 if np.any(enters_conflict) else 0.0

        if np.any(enters_conflict):
            t_enter = float(times[np.argmax(enters_conflict)])
            immediacy_score = self._triangular_score(t_enter, peak=target["ttc_peak"], half_width=target["ttc_half_width"])
        else:
            dist_to_lane = np.maximum(np.abs(future_y) - self.lane_half_width_m, 0.0)
            min_dist = float(np.min(dist_to_lane))
            immediacy_score = np.clip(1.0 - min_dist / 4.0, 0.0, 0.4)

        if np.any(enters_conflict):
            front_x = float(np.min(future_x[enters_conflict]))
            front_proximity = self._triangular_score(front_x, peak=10.0, half_width=8.0)
            ego_lane_conflict_score = float(np.clip(0.75 * ego_lane_conflict_score + 0.25 * front_proximity, 0.0, 1.0))

        notes: List[str] = []
        if roadside_emergence_score < 0.45:
            notes.append("pedestrian is not clearly starting from roadside / curbside region")
        if crossing_direction_score < 0.45:
            notes.append("pedestrian motion is not clearly lateral crossing")
        if ego_lane_conflict_score < 0.45:
            notes.append("predicted pedestrian path does not enter ego lane conflict zone ahead")
        if immediacy_score < 0.35:
            notes.append("crossing severity / TTC does not match the requested tier")

        return {
            "pedestrian_presence_score": float(np.clip(pedestrian_presence_score, 0.0, 1.0)),
            "roadside_emergence_score": float(np.clip(roadside_emergence_score, 0.0, 1.0)),
            "crossing_direction_score": float(np.clip(crossing_direction_score, 0.0, 1.0)),
            "ego_lane_conflict_score": float(np.clip(ego_lane_conflict_score, 0.0, 1.0)),
            "immediacy_score": float(np.clip(immediacy_score, 0.0, 1.0)),
            "notes": notes,
        }

    def _is_crossing_prompt(self, prompt_spec: Any) -> bool:
        if prompt_spec is None:
            return False
        texts: List[str] = []
        for attr in ["raw_prompt", "normalized_prompt", "scenario_type"]:
            if hasattr(prompt_spec, attr):
                value = getattr(prompt_spec, attr)
                if value is not None:
                    texts.append(str(value).lower())
        joined = " ".join(texts)
        keywords = ["横穿", "过马路", "cross", "crossing", "pedestrian"]
        return any(k in joined for k in keywords)

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
