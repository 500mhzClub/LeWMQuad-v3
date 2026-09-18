"""Equivalence at occupied-cell packing and fallback boundaries."""
import numpy as np
import pytest

from lewm.packed_cell_obstacles_development import integer_xy_unique


@pytest.mark.parametrize('dtype', [np.int8, np.int16, np.int32, np.int64, '>i8'])
def test_exact_cells_for_duplicates_negative_coordinates_and_strided_input(dtype):
    rng = np.random.default_rng(2026091604)
    cells = rng.integers(-90, 90, size=(2000, 2)).astype(dtype)
    cells[100:200] = cells[:100]
    for a in (cells, cells[::-2], cells[:0], np.zeros((15, 2), dtype=dtype)):
        original = a.copy()
        expected = np.unique(a, axis=0)
        actual = integer_xy_unique(a, axis=0)
        np.testing.assert_array_equal(actual, expected)
        assert actual.dtype == expected.dtype
        np.testing.assert_array_equal(a, original)


@pytest.mark.parametrize('cells', [
    [[-(2**63), 0], [2**63-1, 0], [0, 0]],
    [[-2**40, -2**40], [2**40, 2**40]],
    [[-2**60, 0], [2**60, 0], [0, 0]],
    [[-2**60-1, 1], [2**60, 0]],
])
def test_large_coordinates_and_key_range_fall_back_without_overflow(cells):
    a = np.array(cells, dtype=np.int64)
    np.testing.assert_array_equal(integer_xy_unique(a, axis=0), np.unique(a, axis=0))


def test_numpy_api_fallback_and_global_function_unchanged():
    original = np.unique
    a = np.array([[2, -3], [2, -3], [-4, 7]])
    actual = integer_xy_unique(a, axis=0, return_index=True, return_inverse=True, return_counts=True)
    expected = original(a, axis=0, return_index=True, return_inverse=True, return_counts=True)
    for x, y in zip(actual, expected):
        np.testing.assert_array_equal(x, y)
    for x in (a.astype(float), np.abs(a).astype(np.uint64), a.tolist(), a[:, 0]):
        np.testing.assert_array_equal(integer_xy_unique(x, axis=0), original(x, axis=0))
    np.testing.assert_array_equal(integer_xy_unique(a), original(a))
    assert np.unique is original
