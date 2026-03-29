from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class PromptSpec:
    """
    Prompt spec aligned with the simplified scene type:
        "突发的行人横穿马路"

    This version adds severity tiers:
        - mild
        - moderate
        - aggressive

    The fields are kept backward-compatible so existing pipeline code can still
    serialize / consume the object without changes.
    """

    raw_prompt: str
    normalized_prompt: str
    scenario_type: str = "generic"
    city: Optional[str] = None
    map_id: Optional[int] = None

    # Backward-compatible legacy fields; crossing semantics usually leaves them neutral.
    occluder_type: str = "none"
    side: str = "auto"
    moderate_traffic: bool = False
    yielding: bool = False
    blind_spot: bool = False
    pedestrian_emerge: bool = False
    use_existing_occluder_first: bool = False
    insert_occluder_if_missing: bool = False

    # Shared editable parameters.
    pedestrian_speed: float = 1.6
    occluder_distance_m: float = 0.0
    occluder_lateral_offset_m: float = 0.0
    prune_conflict_radius_m: float = 0.0
    slow_vehicle_radius_m: float = 0.0
    conflict_point_x_m: float = 12.0
    conflict_point_y_m: float = 0.0

    # New crossing-specific fields.
    severity_level: str = "moderate"  # mild / moderate / aggressive
    ttc_min_s: float = 2.0
    ttc_max_s: float = 3.0
    spawn_from_roadside: bool = True
    keep_scene_minimal: bool = True
    crossing_style: str = "pedestrian_crossing"

    matched_rules: List[str] = field(default_factory=list)
    debug: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class SceneEditROI:
    x_min: float
    y_min: float
    x_max: float
    y_max: float
    tag: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "x_min": self.x_min,
            "y_min": self.y_min,
            "x_max": self.x_max,
            "y_max": self.y_max,
            "tag": self.tag,
        }


@dataclass
class SceneEditResult:
    prompt_spec: PromptSpec
    occluder_source: str
    occluder_index: int
    pedestrian_index: int
    conflict_point_xy: List[float]
    preserved_rois: List[SceneEditROI] = field(default_factory=list)
    removed_vehicle_indices: List[int] = field(default_factory=list)
    slowed_vehicle_indices: List[int] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "prompt_spec": self.prompt_spec.to_dict(),
            "occluder_source": self.occluder_source,
            "occluder_index": self.occluder_index,
            "pedestrian_index": self.pedestrian_index,
            "conflict_point_xy": self.conflict_point_xy,
            "preserved_rois": [roi.to_dict() for roi in self.preserved_rois],
            "removed_vehicle_indices": self.removed_vehicle_indices,
            "slowed_vehicle_indices": self.slowed_vehicle_indices,
            "notes": self.notes,
        }
