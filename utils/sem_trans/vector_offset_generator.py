from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from utils.sem_trans.schema import (
    AnchorMatch,
    BBox7D,
    EditableScene,
    SemanticIntent,
    SemanticPolyline,
    SemTransConfig,
    resample_polyline,
    translate_in_heading_frame,
    wrap_angle,
)


class VectorOffsetGenerator:
    """根据解析后的语义意图，对原始向量场景做“精准手术”。"""

    def __init__(self, config: Optional[SemTransConfig] = None) -> None:
        self._config = config or SemTransConfig()

    def apply(
        self,
        scene: EditableScene,
        intent: SemanticIntent,
        anchor: AnchorMatch,
    ) -> Tuple[EditableScene, Dict[str, List[object]]]:
        edited_scene = scene.clone()
        additions: Dict[str, List[object]] = {
            "lines": [],
            "vehicles": [],
            "pedestrians": [],
            "static_objects": [],
        }

        if intent.map_global.intersection_complexity_delta > 0.0:
            extra_lines = self._augment_intersection_complexity(anchor, intent.map_global.intersection_complexity_delta)
            edited_scene.lines.extend(extra_lines)
            additions["lines"].extend(extra_lines)

        if intent.agent_density.vehicle_count_scale > 1.0:
            extra_vehicles = self._increase_vehicle_density(
                edited_scene,
                anchor,
                intent.agent_density.vehicle_count_scale,
            )
            edited_scene.vehicles.extend(extra_vehicles)
            additions["vehicles"].extend(extra_vehicles)

        if intent.trigger_event is not None and intent.trigger_event.event_type == "Pedestrian_Emergence":
            truck_box, pedestrian_box = self._inject_ghost_probe(intent)
            edited_scene.static_objects.append(truck_box)
            edited_scene.pedestrians.append(pedestrian_box)
            additions["static_objects"].append(truck_box)
            additions["pedestrians"].append(pedestrian_box)

        return edited_scene, additions

    def _augment_intersection_complexity(
        self,
        anchor: AnchorMatch,
        complexity_delta: float,
    ) -> List[SemanticPolyline]:
        """
        在锚点附近注入额外交叉折线。

        为了避免生硬的直线，这里使用一个非常轻量的参数化曲线:
            x = s
            y = A * sin(pi * s / (L / 2))
        再绕 anchor.heading + offset 旋转到局部坐标系。
        """
        num_new_lines = int(np.clip(np.ceil(complexity_delta), 2, self._config.synthetic_lane_count))
        heading_offsets = [-np.pi / 4, np.pi / 4, np.pi / 2]
        new_lines: List[SemanticPolyline] = []

        for index in range(num_new_lines):
            target_heading = wrap_angle(anchor.heading + heading_offsets[index])
            curve_sign = -1.0 if index % 2 == 0 else 1.0
            curve_amplitude = curve_sign * self._config.synthetic_lane_curve_amplitude
            polyline = self._build_crossing_polyline(
                center=anchor.point,
                heading=target_heading,
                total_length=self._config.synthetic_lane_length,
                curve_amplitude=curve_amplitude,
                step=self._config.lane_sample_step,
                tag="synthetic_intersection_lane",
            )
            new_lines.append(polyline)

        return new_lines

    def _build_crossing_polyline(
        self,
        center: np.ndarray,
        heading: float,
        total_length: float,
        curve_amplitude: float,
        step: float,
        tag: str,
    ) -> SemanticPolyline:
        half_length = 0.5 * total_length
        longitudinal = np.arange(-half_length, half_length + 0.5 * step, step, dtype=np.float64)
        normalized = longitudinal / max(half_length, 1e-6)
        lateral = curve_amplitude * np.sin(np.pi * normalized)
        local_points = np.stack([longitudinal, lateral], axis=-1)

        c, s = np.cos(heading), np.sin(heading)
        rotation = np.array([[c, -s], [s, c]], dtype=np.float64)
        world_points = center + local_points @ rotation.T
        world_points = resample_polyline(world_points, step=step)
        return SemanticPolyline(points=world_points, tag=tag, is_new=True)

    def _increase_vehicle_density(
        self,
        scene: EditableScene,
        anchor: AnchorMatch,
        vehicle_count_scale: float,
    ) -> List[BBox7D]:
        base_vehicle_count = max(len(scene.vehicles), 2)
        target_vehicle_count = int(np.ceil(base_vehicle_count * vehicle_count_scale))
        num_to_add = int(np.clip(target_vehicle_count - len(scene.vehicles), 0, self._config.max_density_injections))
        if num_to_add == 0:
            return []

        candidate_headings = self._collect_candidate_headings(scene.lines, anchor)
        median_speed = self._estimate_reference_speed(scene)
        new_vehicles: List[BBox7D] = []

        for index in range(num_to_add):
            heading = candidate_headings[index % len(candidate_headings)]
            ring_index = index // len(candidate_headings)
            lon = 4.0 + ring_index * self._config.vehicle_density_spawn_spacing
            lat = self._config.vehicle_density_lane_offset * (1.0 if index % 2 == 0 else -1.0)
            center = translate_in_heading_frame(anchor.point, heading, lon, lat)

            speed = median_speed
            new_vehicles.append(
                BBox7D(
                    x=float(center[0]),
                    y=float(center[1]),
                    heading=float(heading),
                    length=self._config.default_vehicle_length,
                    width=self._config.default_vehicle_width,
                    vx=float(speed * np.cos(heading)),
                    vy=float(speed * np.sin(heading)),
                    category="vehicle",
                    is_new=True,
                    tag="density_injection",
                )
            )

        return new_vehicles

    def _collect_candidate_headings(
        self,
        lines: Sequence[SemanticPolyline],
        anchor: AnchorMatch,
    ) -> List[float]:
        headings: List[Tuple[float, float]] = []

        for line in lines:
            points = np.asarray(line.points, dtype=np.float64)
            if len(points) < 2:
                continue

            for index in range(len(points) - 1):
                p0, p1 = points[index], points[index + 1]
                midpoint = 0.5 * (p0 + p1)
                distance = float(np.linalg.norm(midpoint - anchor.point))
                if distance > anchor.roi_radius:
                    continue
                headings.append((distance, float(np.arctan2(*(p1 - p0)[::-1]))))

        if not headings:
            return [anchor.heading, wrap_angle(anchor.heading + np.pi / 2)]

        headings = sorted(headings, key=lambda item: item[0])[:6]
        unique_headings: List[float] = []
        for _, heading in headings:
            if all(abs(wrap_angle(heading - existing_heading)) > np.deg2rad(20.0) for existing_heading in unique_headings):
                unique_headings.append(heading)

        return unique_headings or [anchor.heading]

    def _estimate_reference_speed(self, scene: EditableScene) -> float:
        if scene.vehicles:
            return float(np.median([vehicle.speed for vehicle in scene.vehicles]))
        ego_speed = float(np.hypot(*scene.ego_velocity_xy))
        if ego_speed > 0.5:
            return ego_speed
        return self._config.default_vehicle_speed

    def _inject_ghost_probe(self, intent: SemanticIntent) -> Tuple[BBox7D, BBox7D]:
        assert intent.trigger_event is not None
        lon, lat = intent.trigger_event.relative_pos

        # raw 场景是 ego 局部坐标，所以 ego 当前朝向可以直接看作 heading=0。
        ego_heading = 0.0
        truck_center = translate_in_heading_frame(np.zeros(2, dtype=np.float64), ego_heading, lon, lat)
        truck_heading = ego_heading

        truck_box = BBox7D(
            x=float(truck_center[0]),
            y=float(truck_center[1]),
            heading=truck_heading,
            length=self._config.ghost_truck_length,
            width=self._config.ghost_truck_width,
            vx=0.0,
            vy=0.0,
            category="static_object",
            is_new=True,
            tag="ghost_probe_truck",
        )

        ped_center = translate_in_heading_frame(
            truck_center,
            truck_heading,
            lon=-(0.5 * self._config.ghost_truck_length + self._config.ghost_ped_back_offset),
            lat=0.0,
        )

        # 行人速度设置为垂直于 ego 路径，并朝道路中心移动。
        lateral_direction = -1.0 if lat >= 0.0 else 1.0
        ped_vx = 0.0
        ped_vy = lateral_direction * self._config.ghost_ped_speed
        ped_heading = float(np.arctan2(ped_vy, ped_vx))

        pedestrian_box = BBox7D(
            x=float(ped_center[0]),
            y=float(ped_center[1]),
            heading=ped_heading,
            length=self._config.ghost_ped_length,
            width=self._config.ghost_ped_width,
            vx=ped_vx,
            vy=ped_vy,
            category="pedestrian",
            is_new=True,
            tag="ghost_probe_pedestrian",
        )

        return truck_box, pedestrian_box
