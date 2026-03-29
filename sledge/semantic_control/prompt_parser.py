from __future__ import annotations

import re
from typing import Dict, List, Optional, Tuple

from sledge.autoencoder.preprocessing.features.map_id_feature import MAP_NAME_TO_ID
from sledge.semantic_control.prompt_spec import PromptSpec


_CITY_PATTERNS: List[Tuple[str, int, List[str]]] = [
    ("us-ma-boston", MAP_NAME_TO_ID["us-ma-boston"], ["boston", "波士顿"]),
    ("us-nv-las-vegas-strip", MAP_NAME_TO_ID["us-nv-las-vegas-strip"], ["las vegas", "lasvegas", "拉斯维加斯"]),
    ("us-pa-pittsburgh-hazelwood", MAP_NAME_TO_ID["us-pa-pittsburgh-hazelwood"], ["pittsburgh", "匹兹堡"]),
    ("sg-one-north", MAP_NAME_TO_ID["sg-one-north"], ["singapore", "新加坡"]),
]


_SEVERITY_TABLE = {
    "mild": {"ttc_min_s": 3.0, "ttc_max_s": 4.2, "pedestrian_speed": 1.3},
    "moderate": {"ttc_min_s": 2.0, "ttc_max_s": 3.0, "pedestrian_speed": 1.6},
    "aggressive": {"ttc_min_s": 1.2, "ttc_max_s": 2.0, "pedestrian_speed": 1.9},
}


class NaturalLanguagePromptParser:
    """
    Parser aligned with the simplified scene type:
        "突发的行人横穿马路"

    Adds severity tiers for crossing scenes:
        - mild
        - moderate
        - aggressive
    """

    def normalize(self, text: str) -> str:
        text = text.strip()
        replacements = {
            "“": '"',
            "”": '"',
            "‘": "'",
            "’": "'",
            "，": ",",
            "。": ".",
            "：": ":",
            "；": ";",
            "（": "(",
            "）": ")",
            "【": "[",
            "】": "]",
            "、": ",",
            "\n": " ",
            "\t": " ",
        }
        for src, dst in replacements.items():
            text = text.replace(src, dst)
        text = re.sub(r"\s+", " ", text)

        text = re.sub(r"创建|生成|构建|设置", "create", text, flags=re.IGNORECASE)
        text = re.sub(r"场景", "scene", text, flags=re.IGNORECASE)
        text = re.sub(r"突然|突发", "sudden", text, flags=re.IGNORECASE)
        text = re.sub(r"横穿马路|横穿道路|横穿车道|过马路", "pedestrian crossing", text, flags=re.IGNORECASE)
        text = re.sub(r"轻度|温和|轻微", "mild", text, flags=re.IGNORECASE)
        text = re.sub(r"中等|中度|适中", "moderate", text, flags=re.IGNORECASE)
        text = re.sub(r"激进|强烈|重度|危险", "aggressive", text, flags=re.IGNORECASE)
        return text.strip()

    def parse(self, text: str) -> PromptSpec:
        normalized = self.normalize(text)
        lower = normalized.lower()
        matched_rules: List[str] = []

        city, map_id = self._parse_city(lower)
        if city:
            matched_rules.append(f"city:{city}")

        scenario_type = self._parse_scenario_type(lower)
        matched_rules.append(f"scenario:{scenario_type}")

        side = self._parse_side(lower)
        if side != "auto":
            matched_rules.append(f"side:{side}")

        severity = self._parse_severity(lower)
        matched_rules.append(f"severity:{severity}")
        severity_cfg = _SEVERITY_TABLE[severity]

        pedestrian_emerge = self._contains_any(
            lower,
            [
                "pedestrian",
                "行人",
                "walker",
                "walking person",
                "crossing",
                "walk across",
                "sudden",
                "突然出现",
            ],
        )
        if pedestrian_emerge:
            matched_rules.append("actor:pedestrian")

        crossing_required = scenario_type in {"pedestrian_crossing", "sudden_pedestrian_crossing"}
        if crossing_required:
            matched_rules.append("constraint:crossing")

        moderate_traffic = self._contains_any(
            lower,
            ["moderate traffic", "中等交通", "适中交通", "中等车流", "medium traffic"],
        )
        if moderate_traffic:
            matched_rules.append("traffic:moderate")

        explicit_speed = self._parse_speed(lower, default=severity_cfg["pedestrian_speed"])
        distances = self._parse_distances(lower)

        conflict_x = distances.get("conflict_point_x_m", 12.0)
        conflict_y = distances.get("conflict_point_y_m", 0.0)
        ttc_min_s, ttc_max_s = self._parse_ttc(lower, severity_cfg["ttc_min_s"], severity_cfg["ttc_max_s"])

        spec = PromptSpec(
            raw_prompt=text,
            normalized_prompt=normalized,
            scenario_type=scenario_type,
            city=city,
            map_id=map_id,
            occluder_type="none",
            side=side,
            moderate_traffic=moderate_traffic,
            yielding=False,
            blind_spot=False,
            pedestrian_emerge=pedestrian_emerge,
            use_existing_occluder_first=False,
            insert_occluder_if_missing=False,
            pedestrian_speed=explicit_speed,
            occluder_distance_m=0.0,
            occluder_lateral_offset_m=0.0,
            prune_conflict_radius_m=0.0,
            slow_vehicle_radius_m=0.0,
            conflict_point_x_m=conflict_x,
            conflict_point_y_m=conflict_y,
            severity_level=severity,
            ttc_min_s=ttc_min_s,
            ttc_max_s=ttc_max_s,
            spawn_from_roadside=True,
            keep_scene_minimal=True,
            crossing_style="pedestrian_crossing",
            matched_rules=matched_rules,
            debug={
                "normalized_length": len(normalized),
                "distances": distances,
                "crossing_required": crossing_required,
                "severity_cfg": severity_cfg,
            },
        )
        return spec

    def _parse_city(self, text: str) -> Tuple[Optional[str], Optional[int]]:
        for city_name, map_id, patterns in _CITY_PATTERNS:
            if self._contains_any(text, patterns):
                return city_name, map_id
        return None, None

    def _parse_scenario_type(self, text: str) -> str:
        sudden_crossing_patterns = [
            "pedestrian crossing",
            "sudden pedestrian crossing",
            "突发行人",
            "突然 行人",
            "行人横穿",
            "横穿马路",
            "横穿道路",
            "横穿车道",
            "过马路",
            "jaywalk",
            "crossing",
            "walk across",
        ]
        if self._contains_any(text, sudden_crossing_patterns):
            if self._contains_any(text, ["sudden", "突然", "突发"]):
                return "sudden_pedestrian_crossing"
            return "pedestrian_crossing"
        return "generic"

    def _parse_side(self, text: str) -> str:
        if self._contains_any(text, ["left", "左侧", "左边", "左前方"]):
            return "left"
        if self._contains_any(text, ["right", "右侧", "右边", "右前方"]):
            return "right"
        return "auto"

    def _parse_severity(self, text: str) -> str:
        if self._contains_any(text, ["mild", "轻度", "温和", "轻微"]):
            return "mild"
        if self._contains_any(text, ["aggressive", "激进", "重度", "强烈", "危险"]):
            return "aggressive"
        return "moderate"

    def _parse_distances(self, text: str) -> Dict[str, float]:
        parsed: Dict[str, float] = {}

        front_match = re.search(r"(?:front|前方|ahead)\s*(\d+(?:\.\d+)?)\s*m", text)
        if front_match:
            parsed["conflict_point_x_m"] = float(front_match.group(1))

        center_match = re.search(r"(?:center|中间|中心)\s*(\d+(?:\.\d+)?)\s*m", text)
        if center_match:
            parsed["conflict_point_x_m"] = float(center_match.group(1))

        lateral_match = re.search(r"(?:lateral|旁|侧向|横向)\s*(-?\d+(?:\.\d+)?)\s*m", text)
        if lateral_match:
            parsed["conflict_point_y_m"] = float(lateral_match.group(1))

        return parsed

    def _parse_speed(self, text: str, default: float) -> float:
        speed_match = re.search(r"(?:speed|速度)\s*(\d+(?:\.\d+)?)\s*(?:m/s)?", text)
        if speed_match:
            return float(speed_match.group(1))
        return default

    def _parse_ttc(self, text: str, default_min: float, default_max: float) -> Tuple[float, float]:
        range_match = re.search(r"(?:ttc|time to collision)\s*(\d+(?:\.\d+)?)\s*[-~]\s*(\d+(?:\.\d+)?)", text)
        if range_match:
            a = float(range_match.group(1))
            b = float(range_match.group(2))
            return (min(a, b), max(a, b))
        single_match = re.search(r"(?:ttc|time to collision)\s*(\d+(?:\.\d+)?)", text)
        if single_match:
            center = float(single_match.group(1))
            return (max(0.6, center - 0.4), center + 0.4)
        return default_min, default_max

    @staticmethod
    def _contains_any(text: str, keywords: List[str]) -> bool:
        t = text.lower()
        return any(keyword.lower() in t for keyword in keywords)
