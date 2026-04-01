from typing import List
from pathlib import Path


def find_feature_paths(root_path: Path, feature_name: str) -> List[Path]:
    """
    Simple helper function, collecting all available gzip files in a cache.

    Supports both:
    1. root_path = scenario cache root, containing a "log" directory
    2. root_path = the "log" directory itself

    Expected final structure:
        <scan_root>/<scenario_type>/<token>/<feature_name>.gz

    :param root_path: path of feature cache root or its log subdirectory
    :param feature_name: name of feature, excluding file ending
    :return: list of feature paths without .gz suffix
    """
    root_path = Path(root_path)

    # Support both:
    #   /.../scenario_cache_semantic_check
    # and
    #   /.../scenario_cache_semantic_check/log
    scan_root = root_path / "log" if (root_path / "log").is_dir() else root_path

    if not scan_root.exists() or not scan_root.is_dir():
        raise FileNotFoundError(f"Feature cache directory not found or is not a directory: {scan_root}")

    file_paths: List[Path] = []

    for scenario_type_path in scan_root.iterdir():
        # skip metadata or any plain files
        if scenario_type_path.name == "metadata" or not scenario_type_path.is_dir():
            continue

        for token_path in scenario_type_path.iterdir():
            if not token_path.is_dir():
                continue

            feature_path = token_path / f"{feature_name}.gz"
            if feature_path.is_file():
                # Return path without .gz suffix, consistent with existing caller behavior
                file_paths.append(token_path / feature_name)

    return sorted(file_paths)