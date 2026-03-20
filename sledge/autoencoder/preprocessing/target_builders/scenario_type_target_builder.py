from typing import Type
import numpy as np

from nuplan.planning.scenario_builder.abstract_scenario import AbstractScenario
from nuplan.planning.training.preprocessing.feature_builders.abstract_feature_builder import AbstractModelFeature
from nuplan.planning.training.preprocessing.target_builders.abstract_target_builder import AbstractTargetBuilder
from nuplan.common.maps.maps_datatypes import SemanticMapLayer

from sledge.autoencoder.preprocessing.features.map_id_feature import MapID


# 我们可以复用 MapId 的结构，或者自定义一个特征类。为了保持项目兼容性，这里返回 MapId 结构但存储场景索引。

class ScenarioTypeTargetBuilder(AbstractTargetBuilder):
    """根据场景特征（速度、路口、交通灯）识别场景类型并打标签的构建器"""

    def __init__(self, high_speed_threshold: float = 12.0, medium_speed_threshold: float = 5.0):
        """
        :param high_speed_threshold: 高速判定阈值 (m/s), 12m/s 约等于 43km/h
        :param medium_speed_threshold: 中速判定阈值 (m/s), 5m/s 约等于 18km/h
        """
        self._high_speed_threshold = high_speed_threshold
        self._medium_speed_threshold = medium_speed_threshold

        # 内部映射表（对应你要求的 5 类）
        self._type_to_idx = {
            "high_magnitude_speed": 0,
            "medium_magnitude_speed": 1,
            "traversing_intersection": 2,
            "traversing_traffic_light_intersection": 3,
            "unknown": 4
        }

    @classmethod
    def get_feature_unique_name(cls) -> str:
        """定义特征名称，在 config 中引用此名称"""
        return "scenario_type"

    @classmethod
    def get_feature_type(cls) -> Type[AbstractModelFeature]:
        """返回特征类型"""
        return MapID

    def get_targets(self, scenario: AbstractScenario) -> MapID:
        """核心识别逻辑：遍历 nuPlan 场景并提取标签"""

        # 1. 提取基础信息
        ego_speed = scenario.get_ego_speed_at_iteration(0)

        # 获取当前位置的地图对象（用于判断是否在路口）
        ego_state = scenario.initial_ego_state
        map_api = scenario.map_api

        # 检查路口：查询当前坐标下是否有 INTERSECTION 层
        intersection_objects = map_api.get_proximal_map_objects(
            ego_state.center.point,
            radius=2.0,
            layers=[SemanticMapLayer.INTERSECTION]
        )
        is_at_intersection = len(intersection_objects[SemanticMapLayer.INTERSECTION]) > 0

        # 检查交通灯：获取当前迭代的交通灯状态
        traffic_light_data = list(scenario.get_traffic_light_status_at_iteration(0))
        has_traffic_lights = len(traffic_light_data) > 0

        # 2. 规则引擎判断优先级
        if is_at_intersection and has_traffic_lights:
            label_name = "traversing_traffic_light_intersection"
        elif is_at_intersection:
            label_name = "traversing_intersection"
        elif ego_speed >= self._high_speed_threshold:
            label_name = "high_magnitude_speed"
        elif ego_speed >= self._medium_speed_threshold:
            label_name = "medium_magnitude_speed"
        else:
            label_name = "unknown"

        label_index = self._type_to_idx[label_name]

        # 返回封装好的特征对象
        return MapID(id=np.array([label_index]))