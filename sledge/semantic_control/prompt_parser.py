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

_SCENARIO_KEYWORDS = {
    "sudden_pedestrian_crossing": [
        "pedestrian crossing", "sudden pedestrian crossing", "横穿马路", "横穿道路",
        "横穿车道", "行人横穿", "行人突然冲出", "突发行人", "jaywalk", "walk across"
    ],
    "cut_in": [
        "cut in", "cut-in", "lane change into ego lane", "加塞", "插队", "突然并线",
        "强行并线", "旁车切入", "邻车切入", "车辆加塞"
    ],
    "hard_brake": [
        "hard brake", "sudden braking", "sudden brake", "急刹", "前车急刹", "前车突然减速",
        "lead vehicle braking", "near stop ahead", "前方急停"
    ],
}

_SEVERITY_TABLE = {
    "mild": {
        "ttc_min_s": 3.0, "ttc_max_s": 4.2, "pedestrian_speed": 1.3,
        "target_gap_min_m": 8.0, "target_gap_max_m": 14.0, "target_decel_min_mps2": 2.0, "target_decel_max_mps2": 3.5,
    },
    "moderate": {
        "ttc_min_s": 2.0, "ttc_max_s": 3.0, "pedestrian_speed": 1.6,
        "target_gap_min_m": 5.0, "target_gap_max_m": 10.0, "target_decel_min_mps2": 3.0, "target_decel_max_mps2": 5.0,
    },
    "aggressive": {
        "ttc_min_s": 1.2, "ttc_max_s": 2.0, "pedestrian_speed": 1.9,
        "target_gap_min_m": 3.5, "target_gap_max_m": 7.0, "target_decel_min_mps2": 4.5, "target_decel_max_mps2": 7.0,
    },
}


class NaturalLanguagePromptParser:
    """
    Unified parser for multiple rare driving scenarios:
        - sudden pedestrian crossing
        - vehicle cut-in
        - lead vehicle hard brake
    """

    def normalize(self, text: str) -> str:
        text = text.strip()
        replacements = {
            "“": '"', "”": '"', "‘": "'", "’": "'",
            "，": ",", "。": ".", "：": ":", "；": ";",
            "（": "(", "）": ")", "【": "[", "】": "]",
            "、": ",", "\n": " ", "\t": " ",
        }
        for src, dst in replacements.items():
            text = text.replace(src, dst)
        text = re.sub(r"\s+", " ", text)

        text = re.sub(r"创建|生成|构建|设置", "create", text, flags=re.IGNORECASE)
        text = re.sub(r"场景", "scene", text, flags=re.IGNORECASE)
        text = re.sub(r"突然|突发", "sudden", text, flags=re.IGNORECASE)
        text = re.sub(r"横穿马路|横穿道路|横穿车道|过马路", "pedestrian crossing", text, flags=re.IGNORECASE)
        text = re.sub(r"加塞|插队|强行并线|突然并线|切入", "cut in", text, flags=re.IGNORECASE)
        text = re.sub(r"急刹|急刹车|突然减速|急停", "hard brake", text, flags=re.IGNORECASE)
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

        severity = self._parse_severity(lower)
        matched_rules.append(f"severity:{severity}")
        severity_cfg = _SEVERITY_TABLE[severity]

        side = self._parse_side(lower)
        if side != "auto":
            matched_rules.append(f"side:{side}")

        explicit_speed = self._parse_speed(lower, default=severity_cfg["pedestrian_speed"])
        ttc_min_s, ttc_max_s = self._parse_ttc(lower, severity_cfg["ttc_min_s"], severity_cfg["ttc_max_s"])
        distances = self._parse_distances(lower)

        target_gap_min_m, target_gap_max_m = self._parse_gap(
            lower, severity_cfg["target_gap_min_m"], severity_cfg["target_gap_max_m"]
        )
        decel_min, decel_max = self._parse_decel(
            lower, severity_cfg["target_decel_min_mps2"], severity_cfg["target_decel_max_mps2"]
        )

        primary_actor_type = {
            "sudden_pedestrian_crossing": "pedestrian",
            "cut_in": "vehicle",
            "hard_brake": "lead_vehicle",
        }.get(scenario_type, "none")

        crossing_required = scenario_type == "sudden_pedestrian_crossing"
        if crossing_required:
            matched_rules.append("constraint:crossing")

        if scenario_type == "cut_in":
            matched_rules.append("constraint:merge_into_ego_lane")
        if scenario_type == "hard_brake":
            matched_rules.append("constraint:lead_vehicle_close_and_slow")

        spec = PromptSpec(
            raw_prompt=text,
            normalized_prompt=normalized,
            scenario_type=scenario_type,
            city=city,
            map_id=map_id,
            side=side,
            pedestrian_emerge=scenario_type == "sudden_pedestrian_crossing",
            severity_level=severity,
            primary_actor_type=primary_actor_type,
            pedestrian_speed=explicit_speed,
            vehicle_speed=max(2.0, explicit_speed + 4.0),
            target_speed_mps=max(0.5, explicit_speed),
            ttc_min_s=ttc_min_s,
            ttc_max_s=ttc_max_s,
            conflict_point_x_m=distances.get("conflict_point_x_m", 12.0),
            conflict_point_y_m=distances.get("conflict_point_y_m", 0.0),
            target_gap_min_m=target_gap_min_m,
            target_gap_max_m=target_gap_max_m,
            lead_distance_min_m=target_gap_min_m,
            lead_distance_max_m=target_gap_max_m,
            target_decel_min_mps2=decel_min,
            target_decel_max_mps2=decel_max,
            lateral_speed_min_mps=0.8 if severity == "mild" else 1.1 if severity == "moderate" else 1.5,
            lateral_speed_max_mps=1.6 if severity == "mild" else 2.2 if severity == "moderate" else 2.8,
            relative_speed_min_mps=2.0 if severity == "mild" else 3.0 if severity == "moderate" else 4.0,
            relative_speed_max_mps=4.5 if severity == "mild" else 6.5 if severity == "moderate" else 8.0,
            spawn_from_roadside=True,
            keep_scene_minimal=True,
            allow_actor_insertion=True,
            prefer_existing_actor=False,
            crossing_style="pedestrian_crossing" if scenario_type == "sudden_pedestrian_crossing" else scenario_type,
            matched_rules=matched_rules,
            debug={
                "normalized_length": len(normalized),
                "distances": distances,
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
        for scenario_type, keywords in _SCENARIO_KEYWORDS.items():
            if self._contains_any(text, keywords):
                return scenario_type
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

    def _parse_gap(self, text: str, default_min: float, default_max: float) -> Tuple[float, float]:
        range_match = re.search(r"(?:gap|间距|车距|headway)\s*(\d+(?:\.\d+)?)\s*[-~]\s*(\d+(?:\.\d+)?)", text)
        if range_match:
            a = float(range_match.group(1))
            b = float(range_match.group(2))
            return (min(a, b), max(a, b))
        single_match = re.search(r"(?:gap|间距|车距|headway)\s*(\d+(?:\.\d+)?)", text)
        if single_match:
            center = float(single_match.group(1))
            return (max(2.0, center - 2.0), center + 2.0)
        return default_min, default_max

    def _parse_decel(self, text: str, default_min: float, default_max: float) -> Tuple[float, float]:
        range_match = re.search(r"(?:decel|deceleration|减速度)\s*(\d+(?:\.\d+)?)\s*[-~]\s*(\d+(?:\.\d+)?)", text)
        if range_match:
            a = float(range_match.group(1))
            b = float(range_match.group(2))
            return (min(a, b), max(a, b))
        single_match = re.search(r"(?:decel|deceleration|减速度)\s*(\d+(?:\.\d+)?)", text)
        if single_match:
            center = float(single_match.group(1))
            return (max(1.0, center - 1.0), center + 1.0)
        return default_min, default_max

    @staticmethod
    def _contains_any(text: str, keywords: List[str]) -> bool:
        t = text.lower()
        return any(keyword.lower() in t for keyword in keywords)
