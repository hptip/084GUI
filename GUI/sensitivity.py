"""One-feature-at-a-time sensitivity analysis over the validated domain."""

from typing import Any, Mapping

import numpy as np

from prediction import predict_model
from uq import predict_with_uq


def sensitivity_data(bundle: Mapping[str, Any], model_name: str, feature: str, baseline: Mapping[str, float], points: int = 80) -> dict[str, np.ndarray]:
    low = float(bundle["feature_ranges"][feature]["min"])
    high = float(bundle["feature_ranges"][feature]["max"])
    grid = np.linspace(low, high, points)
    rows = np.repeat(np.asarray([[baseline[name] for name in bundle["feature_order"]]], dtype=float), points, axis=0)
    rows[:, bundle["feature_order"].index(feature)] = grid
    result: dict[str, np.ndarray] = {"x": grid, "prediction": predict_model(bundle, model_name, rows)}
    uq_result = predict_with_uq(bundle, model_name, rows)
    if uq_result is not None:
        for key in ("pi90_lower", "pi90_upper", "pi95_lower", "pi95_upper"):
            result[key] = uq_result[key]
    return result