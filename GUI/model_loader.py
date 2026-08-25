"""Load and validate the exported, non-trainable model bundle."""

import json
from pathlib import Path
from typing import Any


REQUIRED_FEATURE_ORDER = ["D", "t", "L", "fy", "Ea", "fc", "et", "eb"]
MODEL_ORDER = ["RF", "ETR", "DTR"]


def load_bundle(path: str | Path) -> dict[str, Any]:
    """Load the JSON bundle and validate the public inference contract."""
    bundle_path = Path(path)
    if not bundle_path.exists():
        raise FileNotFoundError(f"Model bundle not found: {bundle_path}")
    with bundle_path.open("r", encoding="utf-8") as bundle_file:
        bundle = json.load(bundle_file)

    if bundle.get("feature_order") != REQUIRED_FEATURE_ORDER:
        raise ValueError(
            "Bundle feature order must be [D, t, L, fy, Ea, fc, et, eb]."
        )
    for key in ("base_trees", "feature_ranges", "reported_metrics"):
        if key not in bundle:
            raise ValueError(f"Bundle is missing required key: {key}")
    missing_models = set(MODEL_ORDER) - set(bundle["base_trees"])
    if missing_models:
        raise ValueError(f"Bundle is missing models: {sorted(missing_models)}")
    return bundle