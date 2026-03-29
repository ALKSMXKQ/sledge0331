from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Dict, List, Optional, Sequence, Tuple

import numpy as np
import numpy.typing as npt

if TYPE_CHECKING:
    from shapely.geometry import Polygon

from sledge.autoencoder.preprocessing.features.sledge_vector_feature import (
    AgentIndex,
    EgoIndex,
    SledgeVectorElement,
    SledgeVectorRaw,
    StaticObjectIndex,
)


def wrap_angle(angle: float) -> float:
    """将任意角度规约到 [-pi, pi]。"""
    return float(np.arctan2(np.sin(angle), np.cos(angle)))


def rotation_matrix(angle: float) -> npt.NDArray[np.float64]:
    """标准二维旋转矩阵。"""
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, -s], [s, c]], dtype=np.float64)


def transform_points_global_to_local(
    points: npt.NDArray[np.float64],
    origin_xy: npt.NDArray[np.float64],
    origin_heading: float,
) -> npt.NDArray[np.float64]:
    """
    全局坐标 -> 以 ego 为原点的局部坐标。

    数学形式:
        p_local = R(-theta_0) * (p_global - t_0)
    其中:
        theta_0 为 ego 朝向
        t_0 为 ego 平移
    """
    points = np.asarray(points, dtype=np.float64)
    origin_xy = np.asarray(origin_xy, dtype=np.float64)
    return (points - origin_xy) @ rotation_matrix(-origin_heading).T


def transform_points_local_to_global(
    points: npt.NDArray[np.float64],
    origin_xy: npt.NDArray[np.float64],
    origin_heading: float,
) -> npt.NDArray[np.float64]:
    """局部坐标 -> 全局坐标。"""
    points = np.asarray(points, dtype=np.float64)
    origin_xy = np.asarray(origin_xy, dtype=np.float64)
    return points @ rotation_matrix(origin_heading).T + origin_xy


def transform_vectors_global_to_local(
    vectors: npt.NDArray[np.float64],
    origin_heading: float,
) -> npt.NDArray[np.float64]:
    """
    速度/加速度等向量只需要旋转，不需要平移。

    数学形式:
        v_local = R(-theta_0) * v_global
    """
    vectors = np.asarray(vectors, dtype=np.float64)
    return vectors @ rotation_matrix(-origin_heading).T


def translate_in_heading_frame(
    center_xy: Sequence[float],
    heading: float,
    lon: float,
    lat: float,
) -> npt.NDArray[np.float64]:
    """
    在“沿 heading 前向 / 左向”为基的局部坐标中做平移。

    数学形式:
        p' = p + R(heading) * [lon, lat]^T
    """
    local_offset = np.array([lon, lat], dtype=np.float64)
    return np.asarray(center_xy, dtype=np.float64) + rotation_matrix(heading) @ local_offset


def compute_polyline_headings(points: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """根据离散折线点估计每个点的切向 heading。"""
    points = np.asarray(points, dtype=np.float64)
    if len(points) == 0:
        return np.zeros((0,), dtype=np.float64)
    if len(points) == 1:
        return np.zeros((1,), dtype=np.float64)

    deltas = np.diff(points, axis=0)
    headings = np.arctan2(deltas[:, 1], deltas[:, 0])
    headings = np.concatenate([headings, headings[-1:]], axis=0)
    return headings.astype(np.float64)


def resample_polyline(
    points: npt.NDArray[np.float64],
    step: float,
) -> npt.NDArray[np.float64]:
    """按弧长对折线重采样，方便后续统一插值/碰撞检查。"""
    points = np.asarray(points, dtype=np.float64)
    if len(points) < 2:
        return points.copy()

    segment_lengths = np.linalg.norm(np.diff(points, axis=0), axis=1)
    cumulative = np.concatenate([[0.0], np.cumsum(segment_lengths)], axis=0)
    total_length = cumulative[-1]
    if total_length < 1e-6:
        return points[[0]].copy()

    distances = np.arange(0.0, total_length + 0.5 * step, step, dtype=np.float64)
    x = np.interp(distances, cumulative, points[:, 0])
    y = np.interp(distances, cumulative, points[:, 1])
    return np.stack([x, y], axis=-1)


def pad_polylines_to_raw_tensor(
    polylines: Sequence["SemanticPolyline"],
) -> Tuple[npt.NDArray[np.float32], npt.NDArray[np.bool_]]:
    """将可变长折线列表打包成 SledgeVectorRaw 所需的 `(N, P, 3)` + mask。"""
    if not polylines:
        return (
            np.zeros((0, 0, 3), dtype=np.float32),
            np.zeros((0, 0), dtype=bool),
        )

    max_points = max(len(polyline.points) for polyline in polylines)
    states = np.zeros((len(polylines), max_points, 3), dtype=np.float32)
    mask = np.zeros((len(polylines), max_points), dtype=bool)

    for polyline_index, polyline in enumerate(polylines):
        raw_states = polyline.to_raw_line_states()
        num_points = len(raw_states)
        states[polyline_index, :num_points] = raw_states
        mask[polyline_index, :num_points] = True

    return states, mask


@dataclass
class SemTransConfig:
    """SEM-Trans 独立配置。全部放在外挂模块内，避免污染 SLEDGE 主配置。"""

    lane_sample_step: float = 1.0
    intersection_roi_radius: float = 15.0
    min_crossing_angle_deg: float = 35.0
    synthetic_lane_length: float = 28.0
    synthetic_lane_count: int = 3
    synthetic_lane_curve_amplitude: float = 2.0

    vehicle_density_spawn_spacing: float = 7.0
    vehicle_density_lane_offset: float = 1.8
    default_vehicle_speed: float = 6.0
    default_vehicle_length: float = 4.8
    default_vehicle_width: float = 2.0
    max_density_injections: int = 12

    ghost_probe_relative_pos: Tuple[float, float] = (10.0, 2.0)
    ghost_truck_length: float = 8.5
    ghost_truck_width: float = 2.6
    ghost_ped_length: float = 0.8
    ghost_ped_width: float = 0.6
    ghost_ped_speed: float = 1.6
    ghost_ped_back_offset: float = 1.5

    overlap_push_step: float = 0.5
    overlap_max_iterations: int = 20


@dataclass
class MapGlobalIntent:
    """地图层面的语义编辑意图。"""

    intersection_complexity_delta: float = 0.0


@dataclass
class AgentDensityIntent:
    """动态交通体密度意图。"""

    vehicle_count_scale: float = 1.0


@dataclass
class TriggerEventIntent:
    """局部触发事件，例如鬼探头。"""

    event_type: str
    relative_pos: Tuple[float, float]
    occluder: str


@dataclass
class SemanticIntent:
    """意图解析的统一输出。"""

    instruction: str
    map_global: MapGlobalIntent = field(default_factory=MapGlobalIntent)
    agent_density: AgentDensityIntent = field(default_factory=AgentDensityIntent)
    trigger_event: Optional[TriggerEventIntent] = None

    def to_json_dict(self) -> Dict[str, Dict[str, object]]:
        """为了方便调试，输出接近用户描述的 JSON 结构。"""
        payload: Dict[str, Dict[str, object]] = {}
        if abs(self.map_global.intersection_complexity_delta) > 1e-6:
            payload["Map_Global"] = {
                "Intersection_Complexity": f"{self.map_global.intersection_complexity_delta:+.1f}"
            }
        if abs(self.agent_density.vehicle_count_scale - 1.0) > 1e-6:
            payload["Agent_Density"] = {"Vehicle_Count": f"*{self.agent_density.vehicle_count_scale:.2f}"}
        if self.trigger_event is not None:
            payload["Trigger_Event"] = {
                "Type": self.trigger_event.event_type,
                "Relative_Pos": list(self.trigger_event.relative_pos),
                "Occluder": self.trigger_event.occluder,
            }
        return payload

    @property
    def needs_intersection_anchor(self) -> bool:
        return abs(self.map_global.intersection_complexity_delta) > 1e-6


@dataclass
class SemanticPolyline:
    """内部编辑用的地图折线表示。"""

    points: npt.NDArray[np.float64]
    tag: str = "line"
    is_new: bool = False

    def __post_init__(self) -> None:
        self.points = np.asarray(self.points, dtype=np.float64)
        if self.points.ndim != 2 or self.points.shape[1] != 2:
            raise ValueError("SemanticPolyline.points 必须是形如 (N, 2) 的二维坐标数组。")

    def to_raw_line_states(self) -> npt.NDArray[np.float32]:
        """导出为 SLEDGE raw line 所需的 `(x, y, heading)`。"""
        headings = compute_polyline_headings(self.points)
        states = np.concatenate([self.points, headings[:, None]], axis=-1)
        return states.astype(np.float32)


@dataclass
class BBox7D:
    """
    独立于 SLEDGE 内部 schema 的 7 维 box。

    对外:
        [x, y, heading, length, width, vx, vy]
    对内导出到 SLEDGE raw 时:
        [x, y, heading, width, length, speed]
    """

    x: float
    y: float
    heading: float
    length: float
    width: float
    vx: float
    vy: float
    category: str
    is_new: bool = False
    tag: str = ""

    @property
    def center(self) -> npt.NDArray[np.float64]:
        return np.array([self.x, self.y], dtype=np.float64)

    @property
    def speed(self) -> float:
        return float(np.hypot(self.vx, self.vy))

    def to_agent_state(self) -> npt.NDArray[np.float32]:
        """导出为 Sledge raw agent schema。"""
        return np.array(
            [
                self.x,
                self.y,
                wrap_angle(self.heading),
                self.width,
                self.length,
                self.speed,
            ],
            dtype=np.float32,
        )

    def to_static_state(self) -> npt.NDArray[np.float32]:
        """导出为 Sledge raw static object schema。"""
        return np.array(
            [
                self.x,
                self.y,
                wrap_angle(self.heading),
                self.width,
                self.length,
            ],
            dtype=np.float32,
        )

    def as_box7(self) -> npt.NDArray[np.float32]:
        """导出为用户期望的 7 维语义 box。"""
        return np.array(
            [
                self.x,
                self.y,
                wrap_angle(self.heading),
                self.length,
                self.width,
                self.vx,
                self.vy,
            ],
            dtype=np.float32,
        )

    def to_polygon(self) -> "Polygon":
        """将 2D box 变成多边形，用于重叠检测。"""
        from shapely.geometry import Polygon

        half_length = 0.5 * self.length
        half_width = 0.5 * self.width
        corners_local = np.array(
            [
                [half_length, half_width],
                [half_length, -half_width],
                [-half_length, -half_width],
                [-half_length, half_width],
            ],
            dtype=np.float64,
        )
        corners_world = self.center + corners_local @ rotation_matrix(self.heading).T
        return Polygon(corners_world)

    def copy(self) -> "BBox7D":
        return copy.deepcopy(self)


@dataclass
class EditableScene:
    """SEM-Trans 内部使用的可编辑场景。默认全部处于 ego 局部坐标系。"""

    lines: List[SemanticPolyline]
    vehicles: List[BBox7D]
    pedestrians: List[BBox7D]
    static_objects: List[BBox7D]
    green_lights: List[SemanticPolyline]
    red_lights: List[SemanticPolyline]
    ego_velocity_xy: Tuple[float, float]

    @classmethod
    def from_sledge_vector_raw(cls, raw_scene: SledgeVectorRaw) -> "EditableScene":
        """把 SledgeVectorRaw 转成更方便做语义编辑的内部格式。"""

        def _extract_polylines(element: SledgeVectorElement, tag: str) -> List[SemanticPolyline]:
            polylines: List[SemanticPolyline] = []
            for states, mask in zip(element.states, element.mask):
                valid_states = states[mask]
                if len(valid_states) < 2:
                    continue
                polylines.append(SemanticPolyline(points=valid_states[:, :2], tag=tag, is_new=False))
            return polylines

        def _extract_agents(states: npt.NDArray[np.float32], category: str) -> List[BBox7D]:
            boxes: List[BBox7D] = []
            for state in states:
                heading = float(state[AgentIndex.HEADING])
                speed = float(state[AgentIndex.VELOCITY])
                boxes.append(
                    BBox7D(
                        x=float(state[AgentIndex.X]),
                        y=float(state[AgentIndex.Y]),
                        heading=heading,
                        length=float(state[AgentIndex.LENGTH]),
                        width=float(state[AgentIndex.WIDTH]),
                        vx=float(speed * np.cos(heading)),
                        vy=float(speed * np.sin(heading)),
                        category=category,
                        is_new=False,
                    )
                )
            return boxes

        def _extract_static(states: npt.NDArray[np.float32]) -> List[BBox7D]:
            boxes: List[BBox7D] = []
            for state in states:
                boxes.append(
                    BBox7D(
                        x=float(state[StaticObjectIndex.X]),
                        y=float(state[StaticObjectIndex.Y]),
                        heading=float(state[StaticObjectIndex.HEADING]),
                        length=float(state[StaticObjectIndex.LENGTH]),
                        width=float(state[StaticObjectIndex.WIDTH]),
                        vx=0.0,
                        vy=0.0,
                        category="static_object",
                        is_new=False,
                    )
                )
            return boxes

        ego_states = np.asarray(raw_scene.ego.states, dtype=np.float32).reshape(-1)
        ego_velocity_xy = (
            float(ego_states[EgoIndex.VELOCITY_X]) if len(ego_states) > EgoIndex.VELOCITY_X else 0.0,
            float(ego_states[EgoIndex.VELOCITY_Y]) if len(ego_states) > EgoIndex.VELOCITY_Y else 0.0,
        )

        return cls(
            lines=_extract_polylines(raw_scene.lines, "line"),
            vehicles=_extract_agents(np.asarray(raw_scene.vehicles.states, dtype=np.float32), "vehicle"),
            pedestrians=_extract_agents(np.asarray(raw_scene.pedestrians.states, dtype=np.float32), "pedestrian"),
            static_objects=_extract_static(np.asarray(raw_scene.static_objects.states, dtype=np.float32)),
            green_lights=_extract_polylines(raw_scene.green_lights, "green_light"),
            red_lights=_extract_polylines(raw_scene.red_lights, "red_light"),
            ego_velocity_xy=ego_velocity_xy,
        )

    def to_sledge_vector_raw(self) -> SledgeVectorRaw:
        """把编辑后的场景导回 SLEDGE 兼容的 raw 向量。"""
        line_states, line_mask = pad_polylines_to_raw_tensor(self.lines)
        green_states, green_mask = pad_polylines_to_raw_tensor(self.green_lights)
        red_states, red_mask = pad_polylines_to_raw_tensor(self.red_lights)

        vehicle_states = np.array([box.to_agent_state() for box in self.vehicles], dtype=np.float32)
        pedestrian_states = np.array([box.to_agent_state() for box in self.pedestrians], dtype=np.float32)
        static_states = np.array([box.to_static_state() for box in self.static_objects], dtype=np.float32)

        if len(vehicle_states) == 0:
            vehicle_states = np.zeros((0, AgentIndex.size()), dtype=np.float32)
        if len(pedestrian_states) == 0:
            pedestrian_states = np.zeros((0, AgentIndex.size()), dtype=np.float32)
        if len(static_states) == 0:
            static_states = np.zeros((0, StaticObjectIndex.size()), dtype=np.float32)

        ego_states = np.array(
            [
                self.ego_velocity_xy[0],
                self.ego_velocity_xy[1],
                0.0,
                0.0,
            ],
            dtype=np.float32,
        )

        return SledgeVectorRaw(
            lines=SledgeVectorElement(states=line_states, mask=line_mask),
            vehicles=SledgeVectorElement(states=vehicle_states, mask=np.zeros((len(vehicle_states),), dtype=bool)),
            pedestrians=SledgeVectorElement(
                states=pedestrian_states,
                mask=np.zeros((len(pedestrian_states),), dtype=bool),
            ),
            static_objects=SledgeVectorElement(states=static_states, mask=np.zeros((len(static_states),), dtype=bool)),
            green_lights=SledgeVectorElement(states=green_states, mask=green_mask),
            red_lights=SledgeVectorElement(states=red_states, mask=red_mask),
            ego=SledgeVectorElement(states=ego_states, mask=np.ones((1,), dtype=bool)),
        )

    def clone(self) -> "EditableScene":
        return copy.deepcopy(self)


@dataclass
class AnchorMatch:
    """空间拓扑匹配器输出的锚点。"""

    point: npt.NDArray[np.float64]
    heading: float
    roi_radius: float
    reason: str
    supporting_line_ids: Tuple[int, ...] = ()


@dataclass
class SemTransResult:
    """方便调试/可视化的完整输出。"""

    instruction: str
    intent: SemanticIntent
    anchor: AnchorMatch
    scene: EditableScene
    raw_scene: SledgeVectorRaw
