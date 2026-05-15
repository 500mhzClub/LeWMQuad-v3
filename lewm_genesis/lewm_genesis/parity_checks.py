"""Gazebo/Genesis parity helpers for rollout metric audits."""

from __future__ import annotations

from typing import Any


def compare_scalar_metrics(
    gazebo_metrics: dict[str, float],
    genesis_metrics: dict[str, float],
    tolerances: dict[str, float],
) -> dict[str, Any]:
    """Compare scalar rollout metrics against absolute tolerances."""

    results: dict[str, Any] = {}
    for metric_name, tolerance in tolerances.items():
        gazebo_value = gazebo_metrics.get(metric_name)
        genesis_value = genesis_metrics.get(metric_name)
        if gazebo_value is None or genesis_value is None:
            results[metric_name] = {
                "success": False,
                "reason": "missing metric",
                "gazebo": gazebo_value,
                "genesis": genesis_value,
                "tolerance": tolerance,
            }
            continue
        delta = abs(float(gazebo_value) - float(genesis_value))
        results[metric_name] = {
            "success": delta <= float(tolerance),
            "delta": delta,
            "gazebo": float(gazebo_value),
            "genesis": float(genesis_value),
            "tolerance": float(tolerance),
        }
    return results


def parity_summary(metric_results: dict[str, Any]) -> dict[str, Any]:
    failed = [
        metric_name
        for metric_name, result in metric_results.items()
        if not result.get("success", False)
    ]
    return {
        "success": not failed,
        "failed_metrics": failed,
        "metric_count": len(metric_results),
    }
