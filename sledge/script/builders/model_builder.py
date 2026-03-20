import logging
from typing import List

from hydra.utils import instantiate
from omegaconf import DictConfig

from nuplan.planning.script.builders.utils.utils_type import validate_type
from nuplan.planning.training.preprocessing.target_builders.abstract_target_builder import AbstractTargetBuilder

from sledge.autoencoder.modeling.autoencoder_torch_module_wrapper import AutoencoderTorchModuleWrapper
from sledge.autoencoder.modeling.autoencoder_lightning_module_wrapper import AutoencoderLightningModuleWrapper
from sledge.autoencoder.preprocessing.target_builders.scenario_type_target_builder import ScenarioTypeTargetBuilder

logger = logging.getLogger(__name__)


def build_autoencoder_torch_module_wrapper(cfg: DictConfig) -> AutoencoderTorchModuleWrapper:
    """
    Builds the autoencoder module.
    :param cfg: DictConfig. Configuration that is used to run the experiment.
    :return: Instance of AutoencoderTorchModuleWrapper.
    """
    logger.info("Building AutoencoderTorchModuleWrapper...")
    model = instantiate(cfg.autoencoder_model)
    validate_type(model, AutoencoderTorchModuleWrapper)
    if cfg.autoencoder_checkpoint:
        model = AutoencoderLightningModuleWrapper.load_from_checkpoint(cfg.autoencoder_checkpoint, model=model).model
        logger.info(f"Load from checkpoint {cfg.autoencoder_checkpoint}...DONE!")
    logger.info("Building AutoencoderTorchModuleWrapper...DONE!")

    return model


def build_target_builders(cfg: DictConfig) -> List[AbstractTargetBuilder]:
    """
    Builds the target builders based on configuration.
    :param cfg: DictConfig. Configuration that is used to run the experiment.
    :return: List of AbstractTargetBuilder instances.
    """
    target_builders = []

    for builder_name in cfg.target_builder:
        if builder_name == "map_id":
            from sledge.autoencoder.preprocessing.target_builders.map_id_target_builder import MapIdTargetBuilder
            target_builders.append(MapIdTargetBuilder())

        elif builder_name == "scenario_type":
            # 注册新编写的场景类型构建器
            target_builders.append(ScenarioTypeTargetBuilder())

        else:
            raise ValueError(f"Unknown target_builder name: {builder_name}")

    return target_builders