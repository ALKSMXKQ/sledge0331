from __future__ import annotations

import gzip
import json
import pickle
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np

from sledge.autoencoder.preprocessing.features.sledge_vector_feature import (
    SledgeVector,
    SledgeVectorElement,
    SledgeVectorRaw,
)


class _NumpyJSONEncoder(json.JSONEncoder):
    def default(self, obj: Any) -> Any:
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.integer, np.floating, np.bool_)):
            return obj.item()
        if is_dataclass(obj):
            return asdict(obj)
        return super().default(obj)


RAW_SCENE_KEYS = {
    "lines",
    "vehicles",
    "pedestrians",
    "static_objects",
    "green_lights",
    "red_lights",
    "ego",
}


def resolve_feature_path(path_like: str | Path) -> Path:
    path = Path(path_like)
    if path.suffix == ".gz":
        return path
    return path.with_suffix(".gz")


def load_gz_pickle(path_like: str | Path) -> Any:
    path = resolve_feature_path(path_like)
    with gzip.open(path, "rb") as fp:
        return pickle.load(fp)


def save_gz_pickle(path_like: str | Path, obj: Any) -> Path:
    path = resolve_feature_path(path_like)
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wb") as fp:
        pickle.dump(obj, fp, protocol=pickle.HIGHEST_PROTOCOL)
    return path


def is_raw_scene_dict(obj: Any) -> bool:
    if not isinstance(obj, dict):
        return False
    return RAW_SCENE_KEYS.issubset(set(obj.keys()))


def raw_scene_dict_to_feature(data: Dict[str, Any]) -> SledgeVectorRaw:
    if not is_raw_scene_dict(data):
        raise TypeError(f"Expected dict with keys {sorted(RAW_SCENE_KEYS)}, got {type(data)}")

    return SledgeVectorRaw(
        lines=SledgeVectorElement.deserialize(data["lines"]),
        vehicles=SledgeVectorElement.deserialize(data["vehicles"]),
        pedestrians=SledgeVectorElement.deserialize(data["pedestrians"]),
        static_objects=SledgeVectorElement.deserialize(data["static_objects"]),
        green_lights=SledgeVectorElement.deserialize(data["green_lights"]),
        red_lights=SledgeVectorElement.deserialize(data["red_lights"]),
        ego=SledgeVectorElement.deserialize(data["ego"]),
    )


def feature_to_raw_scene_dict(scene: SledgeVectorRaw | SledgeVector) -> Dict[str, Any]:
    if not isinstance(scene, (SledgeVectorRaw, SledgeVector)):
        raise TypeError(f"Expected SledgeVectorRaw or SledgeVector, got {type(scene)}")

    def _elem_dict(elem: SledgeVectorElement) -> Dict[str, Any]:
        return {
            "states": np.asarray(elem.states).copy(),
            "mask": np.asarray(elem.mask).copy(),
        }

    return {
        "lines": _elem_dict(scene.lines),
        "vehicles": _elem_dict(scene.vehicles),
        "pedestrians": _elem_dict(scene.pedestrians),
        "static_objects": _elem_dict(scene.static_objects),
        "green_lights": _elem_dict(scene.green_lights),
        "red_lights": _elem_dict(scene.red_lights),
        "ego": _elem_dict(scene.ego),
    }


def load_raw_scene(path_like: str | Path) -> Tuple[SledgeVectorRaw, str]:
    obj = load_gz_pickle(path_like)

    if isinstance(obj, SledgeVectorRaw):
        return obj, "feature"

    if isinstance(obj, SledgeVector):
        return SledgeVectorRaw(
            lines=obj.lines,
            vehicles=obj.vehicles,
            pedestrians=obj.pedestrians,
            static_objects=obj.static_objects,
            green_lights=obj.green_lights,
            red_lights=obj.red_lights,
            ego=obj.ego,
        ), "feature"

    if is_raw_scene_dict(obj):
        return raw_scene_dict_to_feature(obj), "dict"

    raise TypeError(f"Unsupported raw scene payload type: {type(obj)}")


def save_raw_scene(path_like: str | Path, scene: SledgeVectorRaw | SledgeVector, source_format: str = "dict") -> Path:
    if source_format == "feature":
        return save_gz_pickle(path_like, scene)
    if source_format == "dict":
        return save_gz_pickle(path_like, feature_to_raw_scene_dict(scene))
    raise ValueError(f"Unknown source_format: {source_format}")


def save_json(path_like: str | Path, payload: Any) -> Path:
    path = Path(path_like)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fp:
        json.dump(payload, fp, ensure_ascii=False, indent=2, cls=_NumpyJSONEncoder)
    return path