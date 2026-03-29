from __future__ import annotations

from typing import List, Optional, Sequence

import numpy as np

try:
    from shapely.geometry import Polygon
    from shapely.ops import nearest_points
except ImportError:
    Polygon = object  # type: ignore[assignment]
    nearest_points = None  # type: ignore[assignment]

from utils.sem_trans.schema import AnchorMatch, BBox7D, EditableScene, SemTransConfig


class LatentConsistencyChecker:
    """
    对手工编辑后的场景做“潜在空间一致性修复”。

    这里没有触碰 DiT / RVAE 潜变量本身，而是在进入 RSI/RLM 前，把明显的几何冲突先修平。
    """

    def __init__(self, config: Optional[SemTransConfig] = None) -> None:
        self._config = config or SemTransConfig()

    def refine_scene(
        self,
        scene: EditableScene,
        anchor: AnchorMatch,
        extra_occupied_polygons: Optional[Sequence[Polygon]] = None,
    ) -> EditableScene:
        if nearest_points is None:
            raise ImportError("SEM-Trans consistency checking requires shapely. Please install shapely to enable overlap repair.")

        refined_scene = scene.clone()
        extra_occupied_polygons = list(extra_occupied_polygons or [])

        refined_scene.static_objects = self._refine_box_group(
            refined_scene.static_objects,
            refined_scene.vehicles,
            refined_scene.pedestrians,
            extra_occupied_polygons,
            anchor,
        )
        refined_scene.vehicles = self._refine_box_group(
            refined_scene.vehicles,
            refined_scene.static_objects,
            refined_scene.pedestrians,
            extra_occupied_polygons,
            anchor,
        )
        refined_scene.pedestrians = self._refine_box_group(
            refined_scene.pedestrians,
            refined_scene.static_objects,
            refined_scene.vehicles,
            extra_occupied_polygons,
            anchor,
        )
        return refined_scene

    def _refine_box_group(
        self,
        primary_boxes: List[BBox7D],
        other_group_a: Sequence[BBox7D],
        other_group_b: Sequence[BBox7D],
        extra_occupied_polygons: Sequence[Polygon],
        anchor: AnchorMatch,
    ) -> List[BBox7D]:
        refined_boxes: List[BBox7D] = []

        for box in primary_boxes:
            if not box.is_new:
                refined_boxes.append(box)
                continue

            occupied_polygons = self._collect_occupied_polygons(
                other_group_a=other_group_a,
                other_group_b=other_group_b,
                already_refined=refined_boxes,
                original_group=primary_boxes,
                current_box=box,
                extra_polygons=extra_occupied_polygons,
            )
            refined_box = self._resolve_box_overlap(box, occupied_polygons, anchor.heading)
            refined_boxes.append(refined_box)

        return refined_boxes

    def _collect_occupied_polygons(
        self,
        other_group_a: Sequence[BBox7D],
        other_group_b: Sequence[BBox7D],
        already_refined: Sequence[BBox7D],
        original_group: Sequence[BBox7D],
        current_box: BBox7D,
        extra_polygons: Sequence[Polygon],
    ) -> List[Polygon]:
        occupied: List[Polygon] = [box.to_polygon() for box in other_group_a]
        occupied.extend(box.to_polygon() for box in other_group_b)
        occupied.extend(box.to_polygon() for box in already_refined)
        occupied.extend(box.to_polygon() for box in original_group if box is not current_box and not box.is_new)
        occupied.extend(extra_polygons)
        return occupied

    def _resolve_box_overlap(
        self,
        box: BBox7D,
        occupied_polygons: Sequence[Polygon],
        anchor_heading: float,
    ) -> BBox7D:
        candidate = box.copy()

        for _ in range(self._config.overlap_max_iterations):
            current_polygon = candidate.to_polygon()
            overlaps = [polygon for polygon in occupied_polygons if current_polygon.intersects(polygon)]
            if not overlaps:
                return candidate

            push_vector = np.zeros(2, dtype=np.float64)
            for obstacle in overlaps:
                nearest_on_box, nearest_on_obstacle = nearest_points(current_polygon, obstacle)
                delta = np.array(
                    [
                        nearest_on_box.x - nearest_on_obstacle.x,
                        nearest_on_box.y - nearest_on_obstacle.y,
                    ],
                    dtype=np.float64,
                )

                if np.linalg.norm(delta) < 1e-6:
                    obstacle_centroid = np.array(obstacle.centroid.coords[0], dtype=np.float64)
                    delta = candidate.center - obstacle_centroid

                if np.linalg.norm(delta) < 1e-6:
                    # 如果真的完全重合，就沿 box 法线做一个最小扰动。
                    delta = np.array(
                        [
                            -np.sin(anchor_heading),
                            np.cos(anchor_heading),
                        ],
                        dtype=np.float64,
                    )

                push_vector += delta / max(np.linalg.norm(delta), 1e-6)

            preferred_normal = np.array(
                [
                    -np.sin(anchor_heading),
                    np.cos(anchor_heading),
                ],
                dtype=np.float64,
            )
            push_vector += 0.35 * preferred_normal
            push_norm = max(np.linalg.norm(push_vector), 1e-6)
            push_direction = push_vector / push_norm

            candidate.x += float(self._config.overlap_push_step * push_direction[0])
            candidate.y += float(self._config.overlap_push_step * push_direction[1])

        return candidate
