"""Small, in-memory checks for prospective Go2 development interfaces.

No file access, model loading, simulation, or data discovery occurs here. These
checks do not modify or replace any frozen experiment's validation rules.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from numbers import Integral

import numpy as np


class InterfaceError(ValueError):
    """An input violates the explicitly requested interface."""


def _finite_array(value, *, label: str) -> np.ndarray:
    try:
        array = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise InterfaceError(f"{label}: expected numeric values") from exc
    if not np.isfinite(array).all():
        raise InterfaceError(f"{label}: nonfinite values")
    return array


def validate_v03_manifest(manifest: Mapping) -> int:
    """Reject alternate/ambiguous geometry before calling the legacy builder.

    An explicitly empty scene is valid; silently omitted geometry is not.
    Returns the number of boxes the historical builder should receive.
    """
    if not isinstance(manifest, Mapping):
        raise InterfaceError("manifest: expected a mapping")
    if "objects" in manifest:
        raise InterfaceError("objects schema is not the v03 walls/obstacles/landmarks schema")
    count = 0
    for category in ("walls", "obstacles", "landmarks"):
        rows = manifest.get(category)
        if not isinstance(rows, list):
            raise InterfaceError(f"{category}: an explicit list is required")
        for row in rows:
            if not isinstance(row, Mapping):
                raise InterfaceError(f"{category}: expected object mappings")
            for field in ("center_xyz_m", "size_xyz_m"):
                vector = _finite_array(row.get(field), label=field)
                if vector.shape != (3,):
                    raise InterfaceError(f"{field}: expected three coordinates")
                if field == "size_xyz_m" and not (vector > 0).all():
                    raise InterfaceError("size_xyz_m: dimensions must be positive")
            for angle in ("roll_rad", "pitch_rad", "yaw_rad"):
                value = _finite_array(row.get(angle, 0.0), label=angle)
                if value.shape != ():
                    raise InterfaceError(f"{angle}: expected a scalar")
            count += 1
    return count


def _timestamp(value, label: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral) or value < 0:
        raise InterfaceError(f"{label}: expected a nonnegative integer in nanoseconds")
    return int(value)


def validate_causal_history(
    timestamps_ns: Sequence[int],
    episodes: Sequence[tuple[int, int, int]],
    *,
    image_timestamp_ns: int,
    image_episode: tuple[int, int, int],
    expected_period_ns: int | None = None,
) -> None:
    """Require ordered past samples in the same (env, episode, reset) identity."""
    image_time = _timestamp(image_timestamp_ns, "image_timestamp_ns")
    if len(timestamps_ns) == 0 or len(timestamps_ns) != len(episodes):
        raise InterfaceError("history: empty or mismatched timestamps/episodes")
    if len(image_episode) != 3 or any(
        len(episode) != 3 or tuple(episode) != tuple(image_episode) for episode in episodes
    ):
        raise InterfaceError("history crosses environment, episode, or reset boundary")
    times = [_timestamp(value, "sample timestamp") for value in timestamps_ns]
    if any(right <= left for left, right in zip(times, times[1:])):
        raise InterfaceError("history timestamps must be strictly increasing")
    if times[-1] > image_time:
        raise InterfaceError("history contains a future measurement")
    if expected_period_ns is not None:
        period = _timestamp(expected_period_ns, "expected_period_ns")
        if period == 0 or any(b - a != period for a, b in zip(times, times[1:])):
            raise InterfaceError("history does not match the declared sample period")


def validate_command_match(expected, observed, *, atol: float = 1e-6) -> None:
    """Compare full [ticks, vx/vy/yaw] tapes without zip truncation or NaNs."""
    if not np.isfinite(atol) or atol < 0:
        raise InterfaceError("command tolerance must be finite and nonnegative")
    reference = _finite_array(expected, label="expected commands")
    actual = _finite_array(observed, label="observed commands")
    if reference.ndim != 2 or reference.shape[0] == 0 or reference.shape[1] != 3:
        raise InterfaceError("expected commands must have nonempty shape [ticks, 3]")
    if actual.shape != reference.shape:
        raise InterfaceError("observed command shape differs from the complete expected tape")
    if not np.allclose(reference, actual, atol=atol, rtol=0.0):
        raise InterfaceError("applied commands differ from the expected post-slew tape")


def sensor_channel_summary(values, valid, names: Sequence[str]) -> list[dict]:
    """Describe available samples; a stationary/constant channel is not a failure.

    Validity is explicit per sample/channel. Invalid NaNs and zeros are ignored;
    a nonfinite value marked valid is reported as NONFINITE_VALID. Observed
    variation alone does not establish correct units, calibration, or usefulness.
    """
    data = np.asarray(values, dtype=np.float64)
    mask = np.asarray(valid)
    if data.ndim != 2 or mask.shape != data.shape or mask.dtype != np.bool_:
        raise InterfaceError("sensor values and boolean validity must have shape [time, channel]")
    if len(names) != data.shape[1] or len(set(names)) != len(names):
        raise InterfaceError("sensor names must be unique and match the channel dimension")
    result = []
    for column, name in enumerate(names):
        samples = data[mask[:, column], column]
        count = int(samples.size)
        if count == 0:
            status = "UNAVAILABLE"
        elif not np.isfinite(samples).all():
            status = "NONFINITE_VALID"
        elif count < 2:
            status = "INSUFFICIENT_SAMPLES"
        elif np.all(samples == samples[0]):
            status = "CONSTANT"
        else:
            status = "VARIABLE"
        result.append({"channel": name, "valid_samples": count, "status": status})
    return result


def validate_rigid_transform(transform, *, atol: float = 1e-6) -> None:
    """Require an SE(3) transform; reflected camera bases must be labeled separately."""
    if not np.isfinite(atol) or not 0 <= atol < 1:
        raise InterfaceError("transform tolerance must be finite and in [0, 1)")
    matrix = _finite_array(transform, label="transform")
    if matrix.shape != (4, 4):
        raise InterfaceError("transform must have shape [4, 4]")
    if not np.allclose(matrix[3], [0, 0, 0, 1], atol=atol, rtol=0):
        raise InterfaceError("transform has an invalid homogeneous row")
    rotation = matrix[:3, :3]
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=atol, rtol=0):
        raise InterfaceError("transform basis is not orthonormal")
    if not np.isclose(np.linalg.det(rotation), 1.0, atol=atol, rtol=0):
        raise InterfaceError("transform is a reflection, not a proper rotation")
