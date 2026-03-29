from __future__ import annotations

import re
from typing import Optional, Tuple

from utils.sem_trans.schema import (
    AgentDensityIntent,
    MapGlobalIntent,
    SemanticIntent,
    SemTransConfig,
    TriggerEventIntent,
)


class MockIntentParser:
    """
    用 Mock 规则模拟 LLM 意图解析。

    这样做的目的不是替代真正的 LLM，而是先把下游四个几何/拓扑子模块跑通。
    后续若要换成在线 LLM，只需要替换 `parse()` 的实现，输出仍然保持 SemanticIntent。
    """

    def __init__(self, config: Optional[SemTransConfig] = None) -> None:
        self._config = config or SemTransConfig()

    def parse(self, instruction: str) -> SemanticIntent:
        map_intent = MapGlobalIntent(
            intersection_complexity_delta=self._parse_intersection_complexity(instruction)
        )
        density_intent = AgentDensityIntent(vehicle_count_scale=self._parse_vehicle_density(instruction))
        trigger_event = self._parse_trigger_event(instruction)

        return SemanticIntent(
            instruction=instruction,
            map_global=map_intent,
            agent_density=density_intent,
            trigger_event=trigger_event,
        )

    def _parse_intersection_complexity(self, instruction: str) -> float:
        if "复杂路口" in instruction or "复杂交叉口" in instruction:
            return 2.0
        if "路口" in instruction or "交叉口" in instruction:
            return 1.0
        return 0.0

    def _parse_vehicle_density(self, instruction: str) -> float:
        """
        兼容几类常见说法:
        - 车流增加
        - 车流增加 50%
        - 车流增加到 2 倍
        """
        if "车流" not in instruction and "车多" not in instruction and "拥堵" not in instruction:
            return 1.0

        ratio_match = re.search(r"(?:车流|车流量).*?(?:增加到|提升到|变成)\s*([0-9.]+)\s*倍", instruction)
        if ratio_match:
            return max(1.0, float(ratio_match.group(1)))

        percent_match = re.search(r"(?:车流|车流量).*?(?:增加|提升)\s*([0-9.]+)\s*%", instruction)
        if percent_match:
            return 1.0 + float(percent_match.group(1)) / 100.0

        if "车流增加" in instruction or "车多" in instruction:
            return 1.5

        if "拥堵" in instruction:
            return 1.3

        return 1.0

    def _parse_trigger_event(self, instruction: str) -> Optional[TriggerEventIntent]:
        if "鬼探头" not in instruction and "探头" not in instruction:
            return None

        occluder = "Truck" if ("货车" in instruction or "卡车" in instruction) else "StaticObject"
        relative_pos = self._parse_relative_pos(instruction)
        return TriggerEventIntent(
            event_type="Pedestrian_Emergence",
            relative_pos=relative_pos,
            occluder=occluder,
        )

    def _parse_relative_pos(self, instruction: str) -> Tuple[float, float]:
        """
        解析相对 ego 的位置。

        约定:
        - x: 前向为正
        - y: 左侧为正、右侧为负
        """
        default_lon, default_lat = self._config.ghost_probe_relative_pos

        lon_match = re.search(r"前方\s*([0-9.]+)\s*米", instruction)
        lon = float(lon_match.group(1)) if lon_match else default_lon

        lat_match = re.search(r"(左侧|右侧|侧方)\s*([0-9.]+)\s*米", instruction)
        if lat_match:
            side, value_text = lat_match.groups()
            value = float(value_text)
            if side == "右侧":
                lat = -abs(value)
            else:
                lat = abs(value)
        else:
            lat = default_lat

        return lon, lat
