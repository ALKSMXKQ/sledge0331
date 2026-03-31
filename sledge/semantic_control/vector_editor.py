from __future__ import annotations

from typing import Tuple

from sledge.autoencoder.preprocessing.features.sledge_vector_feature import SledgeVectorRaw
from sledge.semantic_control.editors import CutInEditor, HardBrakeEditor, PedestrianCrossingEditor
from sledge.semantic_control.prompt_spec import PromptSpec, SceneEditResult


class SemanticSceneEditor:
    """
    Unified scene editor dispatcher for multiple rare scenario families.
    """

    def __init__(self) -> None:
        self.crossing_editor = PedestrianCrossingEditor()
        self.cut_in_editor = CutInEditor()
        self.hard_brake_editor = HardBrakeEditor()

    def edit(self, scene: SledgeVectorRaw, spec: PromptSpec) -> Tuple[SledgeVectorRaw, SceneEditResult]:
        if spec.scenario_type == "sudden_pedestrian_crossing":
            return self.crossing_editor.edit(scene, spec)
        if spec.scenario_type == "cut_in":
            return self.cut_in_editor.edit(scene, spec)
        if spec.scenario_type == "hard_brake":
            return self.hard_brake_editor.edit(scene, spec)

        result = SceneEditResult(
            prompt_spec=spec,
            primary_actor_type="none",
            primary_actor_index=-1,
            notes=[f"No editor rule matched scenario_type={spec.scenario_type}; scene left unchanged."],
        )
        return scene, result
