"""Exact inference and domain validation for serialized regression trees."""

from typing import Any, Mapping

import numpy as np


def predict_one_tree(tree: Mapping[str, Any], row: np.ndarray) -> float:
    node = 0
    while True:
        feature = int(tree["feature"][node])
        if feature < 0:
            return float(tree["value"][node])
        threshold = float(tree["threshold"][node])
        node = int(tree["children_left"][node] if row[feature] <= threshold else tree["children_right"][node])


def predict_tree_batch(tree: Mapping[str, Any], rows: np.ndarray) -> np.ndarray:
    return np.asarray([predict_one_tree(tree, row) for row in rows], dtype=float)


def predict_model(bundle: Mapping[str, Any], model_name: str, rows: np.ndarray) -> np.ndarray:
    """Predict with one exported tree or ensemble without retraining."""
    trees = bundle["base_trees"][model_name]
    predictions = np.vstack([predict_tree_batch(tree, rows) for tree in trees])
    return predictions.mean(axis=0)


def predict_all(bundle: Mapping[str, Any], rows: np.ndarray) -> dict[str, np.ndarray]:
    return {model: predict_model(bundle, model, rows) for model in ("RF", "ETR", "DTR")}


def domain_violations(bundle: Mapping[str, Any], values: Mapping[str, float]) -> dict[str, tuple[float, float, float]]:
    violations = {}
    for feature in bundle["feature_order"]:
        value = float(values[feature])
        limits = bundle["feature_ranges"][feature]
        low, high = float(limits["min"]), float(limits["max"])
        if not low <= value <= high:
            violations[feature] = (value, low, high)
    return violations


def validate_domain(bundle: Mapping[str, Any], values: Mapping[str, float]) -> None:
    violations = domain_violations(bundle, values)
    if violations:
        details = ", ".join(f"{feature}={value:g} [{low:g}, {high:g}]" for feature, (value, low, high) in violations.items())
        raise ValueError(f"Input is outside the validated training domain. Prediction is not recommended. {details}")