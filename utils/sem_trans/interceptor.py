from __future__ import annotations

from typing import TYPE_CHECKING, Dict, Optional, Sequence

import numpy as np

if TYPE_CHECKING:
    from shapely.geometry import Polygon

from sledge.autoencoder.preprocessing.features.sledge_vector_feature import SledgeConfig, SledgeVectorRaw
from utils.sem_trans.consistency_checker import LatentConsistencyChecker
from utils.sem_trans.intent_parser import MockIntentParser
from utils.sem_trans.schema import (
    EditableScene,
    SemTransConfig,
    SemTransResult,
    transform_points_global_to_local,
)
from utils.sem_trans.spatial_matcher import SpatialMatcher
from utils.sem_trans.vector_offset_generator import VectorOffsetGenerator


class SEMTransInterceptor:
    """
    SEM-Trans 总入口。

    它是一个外挂式“前置拦截器”:
        原始场景 -> SEM-Trans -> 修改后的 SledgeVectorRaw -> 原有 RSI/RLM -> DiT
    """

    def __init__(
        self,
        sledge_config: Optional[SledgeConfig] = None,
        sem_config: Optional[SemTransConfig] = None,
        intent_parser: Optional[MockIntentParser] = None,
        spatial_matcher: Optional[SpatialMatcher] = None,
        vector_offset_generator: Optional[VectorOffsetGenerator] = None,
        consistency_checker: Optional[LatentConsistencyChecker] = None,
    ) -> None:
        self.sledge_config = sledge_config or SledgeConfig()
        self.sem_config = sem_config or SemTransConfig()

        self.intent_parser = intent_parser or MockIntentParser(self.sem_config)
        self.spatial_matcher = spatial_matcher or SpatialMatcher(self.sem_config)
        self.vector_offset_generator = vector_offset_generator or VectorOffsetGenerator(self.sem_config)
        self.consistency_checker = consistency_checker or LatentConsistencyChecker(self.sem_config)

    def intercept_raw_scene(
        self,
        raw_scene: SledgeVectorRaw,
        instruction: str,
        extra_occupied_polygons: Optional[Sequence[Polygon]] = None,
        return_debug: bool = False,
    ):
        intent = self.intent_parser.parse(instruction)
        editable_scene = EditableScene.from_sledge_vector_raw(raw_scene)
        anchor = self.spatial_matcher.find_anchor(editable_scene, intent)

        edited_scene, _ = self.vector_offset_generator.apply(editable_scene, intent, anchor)
        refined_scene = self.consistency_checker.refine_scene(
            edited_scene,
            anchor,
            extra_occupied_polygons=extra_occupied_polygons,
        )
        refined_raw_scene = refined_scene.to_sledge_vector_raw()

        if return_debug:
            return SemTransResult(
                instruction=instruction,
                intent=intent,
                anchor=anchor,
                scene=refined_scene,
                raw_scene=refined_raw_scene,
            )
        return refined_raw_scene

    def intercept_features_dict(
        self,
        features: Dict[str, object],
        instruction: str,
        extra_occupied_polygons: Optional[Sequence[Polygon]] = None,
    ) -> Dict[str, object]:
        """
        这是最接近训练/缓存阶段“外挂接入”的调用方式:
            features["sledge_raw"] = SEM-Trans(features["sledge_raw"], text)
        """
        if "sledge_raw" not in features:
            raise KeyError("features 中未找到 'sledge_raw'，无法执行 SEM-Trans 前置拦截。")

        updated_features = dict(features)
        updated_features["sledge_raw"] = self.intercept_raw_scene(
            raw_scene=features["sledge_raw"],  # type: ignore[arg-type]
            instruction=instruction,
            extra_occupied_polygons=extra_occupied_polygons,
        )
        return updated_features

    def intercept_scenario(
        self,
        scenario,
        instruction: str,
        return_debug: bool = False,
    ):
        """
        从 nuPlan Scenario 直接构建 raw scene，再进行语义编辑。

        这个入口适合离线缓存或调试，不需要修改任何 SLEDGE 核心代码。
        """
        from sledge.autoencoder.preprocessing.feature_builders.sledge_raw_feature_builder import SledgeRawFeatureBuilder

        raw_builder = SledgeRawFeatureBuilder(self.sledge_config)
        raw_scene = raw_builder.get_features_from_scenario(scenario)
        occupied_polygons = self._collect_local_occupied_polygons(scenario)
        return self.intercept_raw_scene(
            raw_scene=raw_scene,
            instruction=instruction,
            extra_occupied_polygons=occupied_polygons,
            return_debug=return_debug,
        )

    def _collect_local_occupied_polygons(self, scenario) -> Sequence[Polygon]:
        """
        可选地把 map 上的多边形也拉进一致性检查。

        这里的关键点是把全局 map polygon 统一转到 ego 局部坐标:
            p_local = R(-theta_ego) * (p_global - t_ego)
        """
        map_api = scenario.map_api
        ego_state = scenario.initial_ego_state
        ego_origin = np.array([ego_state.center.x, ego_state.center.y], dtype=np.float64)
        ego_heading = float(ego_state.center.heading)

        candidate_layer_names = [
            "ROADBLOCK",
            "INTERSECTION",
            "CROSSWALK",
            "WALKWAYS",
            "CARPARK_AREA",
            "BOUNDARY",
            "BUILDING",
        ]

        semantic_map_layer = __import__(
            "nuplan.common.maps.maps_datatypes",
            fromlist=["SemanticMapLayer"],
        ).SemanticMapLayer

        layers = [
            getattr(semantic_map_layer, layer_name)
            for layer_name in candidate_layer_names
            if hasattr(semantic_map_layer, layer_name)
        ]
        if not layers:
            return []

        proximal_objects = map_api.get_proximal_map_objects(
            point=ego_state.center.point,
            radius=self.sledge_config.radius,
            layers=layers,
        )

        local_polygons = []
        for layer in layers:
            for map_object in proximal_objects[layer]:
                polygon = getattr(map_object, "polygon", None)
                if polygon is None or polygon.is_empty:
                    continue

                exterior = np.asarray(polygon.exterior.coords, dtype=np.float64)
                local_exterior = transform_points_global_to_local(exterior[:, :2], ego_origin, ego_heading)
                from shapely.geometry import Polygon

                local_polygons.append(Polygon(local_exterior))

        return local_polygons
