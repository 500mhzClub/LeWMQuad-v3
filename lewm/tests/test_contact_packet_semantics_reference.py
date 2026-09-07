import math

import numpy as np
import pytest

from scripts.run_contact_packet_semantics_development_v1 import native_force_norm


def test_reference_retains_native_float32_components_without_float32_norm_rounding():
    force = np.array([3.4, 2.6, 1.7], dtype=np.float32)
    packet = {'link_a': np.array([[1]]), 'force_a': force[None, None, :]}
    expected = math.sqrt(math.fsum(float(value) ** 2 for value in force))
    assert abs(float(np.linalg.norm(force)) - expected) > 1e-9
    assert native_force_norm(packet, {1}, 0) == pytest.approx(expected, abs=1e-12, rel=0)


def test_reference_selects_robot_side_and_individual_contact():
    packet = {'link_a': np.array([[10, 1]]),
              'force_a': np.array([[[900, 0, 0], [3, 4, 0]]], dtype=np.float32),
              'force_b': np.array([[[0, -2, 0], [-3, -4, 0]]], dtype=np.float32)}
    assert native_force_norm(packet, {1}, 0) == 2
    assert native_force_norm(packet, {1}, 1) == 5
