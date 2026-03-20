from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, List
import torch
from nuplan.planning.script.builders.utils.utils_type import validate_type
from nuplan.planning.training.preprocessing.feature_builders.abstract_feature_builder import AbstractModelFeature
from nuplan.planning.training.preprocessing.features.abstract_model_feature import FeatureDataType, to_tensor

# === 1. 定义你的场景映射表 ===
# 建议先只列出你关心的核心场景，其他的归为 unknown 或 default
SCENE_NAME_ID_ABBR = [
    (0, "unknown", "UNK"),
    (1, "starting_left_turn", "L_TURN"),
    (2, "starting_right_turn", "R_TURN"),
    (3, "starting_straight_traffic", "STR"),
    (4, "traversing_intersection", "INT"),
    # 你可以在这里添加更多 nuPlan 支持的场景类型
]

SCENE_NAME_TO_ID = {name: id for id, name, abbr in SCENE_NAME_ID_ABBR}
SCENE_ID_TO_NAME = {id: name for id, name, abbr in SCENE_NAME_ID_ABBR}


@dataclass
class SceneType(AbstractModelFeature):
    """
    Feature class to store scene type id.
    完全复刻 MapID 的实现逻辑。
    """
    id: FeatureDataType

    def to_device(self, device: torch.device) -> SceneType:
        validate_type(self.id, torch.Tensor)
        return SceneType(id=self.id.to(device=device))

    def to_feature_tensor(self) -> SceneType:
        return SceneType(id=to_tensor(self.id))

    @classmethod
    def deserialize(cls, data: Dict[str, Any]) -> SceneType:
        return SceneType(id=data["id"])

    def unpack(self) -> List[SceneType]:
        return [SceneType(id) for id in zip(self.id)]
Z
    def torch_to_numpy(self) -> SceneType:
        return SceneType(id=self.id.detach().cpu().numpy())

    # 这一步是为了方便后续从 .gz 读出来时也是标准格式
    def serialize(self) -> Dict[str, Any]:
        return {"id": self.id}