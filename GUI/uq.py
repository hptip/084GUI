"""Prediction intervals reconstructed from exported model-specific UQ data."""

from typing import Any, Mapping

import numpy as np

from prediction import predict_model


UQ_LEVELS = ("90", "95")


def uq_available(bundle: Mapping[str, Any], model_name: str) -> bool:
    """Return true only when every required UQ component exists for a model."""
    required = ("scale_trees", "quantiles", "shrinkage_factors")
    return all(
        model_name in bundle.get(component, {})
        for component in required
    ) and all(
        level in bundle["quantiles"][model_name]
        and level in bundle["shrinkage_factors"][model_name]
        for level in UQ_LEVELS
    )


def _scale_prediction(bundle: Mapping[str, Any], model_name: str, rows: np.ndarray) -> np.ndarray:
    scale_bundle = dict(bundle)
    scale_bundle["base_trees"] = bundle["scale_trees"]
    scale = predict_model(scale_bundle, model_name, rows)
    if bool(bundle.get("use_log", {}).get(model_name, False)):
        scale = np.exp(scale) - float(bundle.get("log_eps", 1.0))
    return np.maximum(scale, 1e-9)


def predict_with_uq(
    bundle: Mapping[str, Any], model_name: str, rows: np.ndarray
) -> dict[str, np.ndarray] | None:
    """Return point prediction, calibrated intervals, and widths for one model."""
    if not uq_available(bundle, model_name):
        return None
    prediction = predict_model(bundle, model_name, rows)
    scale = _scale_prediction(bundle, model_name, rows)
    result: dict[str, np.ndarray] = {
        "prediction": prediction,
        "scale": scale,
    }
    for level in UQ_LEVELS:
        half_width = (
            float(bundle["quantiles"][model_name][level])
            * float(bundle["shrinkage_factors"][model_name][level])
            * scale
        )
        lower = np.maximum(prediction - half_width, 0.0)
        upper = prediction + half_width
        result[f"pi{level}_lower"] = lower
        result[f"pi{level}_upper"] = upper
        result[f"width{level}"] = upper - lower
    return result


def test_uq_bundle(bundle: Mapping[str, Any]) -> None:
    """Validate UQ completeness and interval ordering for all three models."""
    missing = [model for model in ("RF", "ETR", "DTR") if not uq_available(bundle, model)]
    if missing:
        raise ValueError(f"UQ unavailable for model(s): {', '.join(missing)}")

    values = [bundle["feature_ranges"][feature]["median"] for feature in bundle["feature_order"]]
    rows = np.asarray([values], dtype=float)
    for model in ("RF", "ETR", "DTR"):
        result = predict_with_uq(bundle, model, rows)
        assert result is not None
        prediction = result["prediction"]
        for level in UQ_LEVELS:
            lower = result[f"pi{level}_lower"]
            upper = result[f"pi{level}_upper"]
            width = result[f"width{level}"]
            if not np.all(lower <= prediction) or not np.all(prediction <= upper):
                raise ValueError(f"{model} {level}% interval does not contain prediction.")
            if not np.all(width > 0):
                raise ValueError(f"{model} {level}% interval width must be positive.")
        if not np.all(result["width95"] >= result["width90"]):
            raise ValueError(f"{model} 95% interval width is smaller than 90% width.")
