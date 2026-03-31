from __future__ import annotations

from typing import Any

from sledge.autoencoder.preprocessing.features.sledge_vector_feature import SledgeVector
from sledge.semantic_control.evaluators import (
    CrossingAlignmentEvaluator,
    CutInAlignmentEvaluator,
    HardBrakeAlignmentEvaluator,
)


class PromptAlignmentEvaluator:
    """
    Unified evaluator dispatcher for multiple rare scenario families.
    """

    def __init__(self) -> None:
        self.crossing_evaluator = CrossingAlignmentEvaluator()
        self.cut_in_evaluator = CutInAlignmentEvaluator()
        self.braking_evaluator = HardBrakeAlignmentEvaluator()

    def evaluate(self, sledge_vector: SledgeVector, prompt_spec: Any = None):
        scenario_type = getattr(prompt_spec, "scenario_type", "generic") if prompt_spec is not None else "generic"
        if scenario_type == "sudden_pedestrian_crossing":
            return self.crossing_evaluator.evaluate(sledge_vector, prompt_spec)
        if scenario_type == "cut_in":
            return self.cut_in_evaluator.evaluate(sledge_vector, prompt_spec)
        if scenario_type == "hard_brake":
            return self.braking_evaluator.evaluate(sledge_vector, prompt_spec)
        return self.crossing_evaluator.evaluate(sledge_vector, prompt_spec)
