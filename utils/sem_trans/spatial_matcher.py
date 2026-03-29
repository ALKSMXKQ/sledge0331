from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import numpy as np
import numpy.typing as npt

from utils.sem_trans.schema import AnchorMatch, EditableScene, SemanticIntent, SemTransConfig, wrap_angle


class SpatialMatcher:
    """根据文本意图在原始向量场景中寻找最适合编辑的拓扑锚点。"""

    def __init__(self, config: Optional[SemTransConfig] = None) -> None:
        self._config = config or SemTransConfig()
        self._min_cross_angle = np.deg2rad(self._config.min_crossing_angle_deg)

    def find_anchor(self, scene: EditableScene, intent: SemanticIntent) -> AnchorMatch:
        if intent.needs_intersection_anchor:
            intersection_anchor = self._find_intersection_anchor(scene.lines)
            if intersection_anchor is not None:
                return intersection_anchor

        fallback_heading = self._estimate_heading_at_point(scene.lines, np.array([8.0, 0.0], dtype=np.float64))
        return AnchorMatch(
            point=np.array([8.0, 0.0], dtype=np.float64),
            heading=fallback_heading,
            roi_radius=self._config.intersection_roi_radius,
            reason="fallback_forward_roi",
        )

    def _find_intersection_anchor(self, lines: Sequence) -> Optional[AnchorMatch]:
        candidates: List[Tuple[float, npt.NDArray[np.float64], Tuple[int, int]]] = []

        for line_index_a, line_a in enumerate(lines):
            points_a = np.asarray(line_a.points, dtype=np.float64)
            if len(points_a) < 2:
                continue

            for line_index_b in range(line_index_a + 1, len(lines)):
                points_b = np.asarray(lines[line_index_b].points, dtype=np.float64)
                if len(points_b) < 2:
                    continue

                for segment_index_a in range(len(points_a) - 1):
                    p0 = points_a[segment_index_a]
                    p1 = points_a[segment_index_a + 1]
                    heading_a = np.arctan2(*(p1 - p0)[::-1])

                    for segment_index_b in range(len(points_b) - 1):
                        q0 = points_b[segment_index_b]
                        q1 = points_b[segment_index_b + 1]
                        heading_b = np.arctan2(*(q1 - q0)[::-1])

                        angle_gap = abs(wrap_angle(heading_a - heading_b))
                        angle_gap = min(angle_gap, np.pi - angle_gap)
                        if angle_gap < self._min_cross_angle:
                            continue

                        intersection = self._segment_intersection(p0, p1, q0, q1)
                        if intersection is None:
                            continue

                        # 分数越小越好。这里优先选择距离 ego 原点更近的交叉点。
                        score = float(np.linalg.norm(intersection))
                        candidates.append((score, intersection, (line_index_a, line_index_b)))

        if not candidates:
            return None

        _, best_point, supporting_lines = min(candidates, key=lambda item: item[0])
        heading = self._estimate_heading_at_point(lines, best_point)
        return AnchorMatch(
            point=best_point,
            heading=heading,
            roi_radius=self._config.intersection_roi_radius,
            reason="lane_segment_intersection",
            supporting_line_ids=supporting_lines,
        )

    def _segment_intersection(
        self,
        p0: npt.NDArray[np.float64],
        p1: npt.NDArray[np.float64],
        q0: npt.NDArray[np.float64],
        q1: npt.NDArray[np.float64],
    ) -> Optional[npt.NDArray[np.float64]]:
        """
        线段交点的显式数学求解。

        令:
            p(t) = p0 + t * r
            q(u) = q0 + u * s
        当 0<=t<=1 且 0<=u<=1 时，如果 p(t)=q(u)，则交点在线段内部。
        """
        r = p1 - p0
        s = q1 - q0
        denominator = self._cross_2d(r, s)
        qp = q0 - p0

        if abs(denominator) < 1e-8:
            return None

        t = self._cross_2d(qp, s) / denominator
        u = self._cross_2d(qp, r) / denominator
        if 0.0 <= t <= 1.0 and 0.0 <= u <= 1.0:
            return p0 + t * r
        return None

    @staticmethod
    def _cross_2d(a: npt.NDArray[np.float64], b: npt.NDArray[np.float64]) -> float:
        return float(a[0] * b[1] - a[1] * b[0])

    def _estimate_heading_at_point(
        self,
        lines: Sequence,
        point: npt.NDArray[np.float64],
    ) -> float:
        nearby_segments: List[Tuple[float, float]] = []

        for line in lines:
            points = np.asarray(line.points, dtype=np.float64)
            if len(points) < 2:
                continue

            for index in range(len(points) - 1):
                p0, p1 = points[index], points[index + 1]
                midpoint = 0.5 * (p0 + p1)
                distance = float(np.linalg.norm(midpoint - point))
                heading = np.arctan2(*(p1 - p0)[::-1])
                nearby_segments.append((distance, heading))

        if not nearby_segments:
            return 0.0

        nearest_segments = sorted(nearby_segments, key=lambda item: item[0])[:4]
        cos_mean = float(np.mean([np.cos(heading) for _, heading in nearest_segments]))
        sin_mean = float(np.mean([np.sin(heading) for _, heading in nearest_segments]))
        return float(np.arctan2(sin_mean, cos_mean))
