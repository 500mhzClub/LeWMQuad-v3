from __future__ import annotations

import numpy as np
import pytest

from lewm.actions import (
    ACTIVE_BLOCK_DIM,
    ACTIVE_BLOCK_ORDER,
    active_block_metadata,
    active_block_to_matrix,
    assert_active_block_metadata_compatible,
    encode_active_block,
    encode_executed_command_block,
)


def test_encode_active_block_uses_channel_major_order() -> None:
    block = encode_active_block(
        vx_body_mps=[1, 2, 3, 4, 5],
        vy_body_mps=[10, 20, 30, 40, 50],
        yaw_rate_radps=[100, 200, 300, 400, 500],
    )

    np.testing.assert_array_equal(
        block,
        np.array(
            [1, 2, 3, 4, 5, 10, 20, 30, 40, 50, 100, 200, 300, 400, 500],
            dtype=np.float32,
        ),
    )


def test_encode_executed_command_block_payload() -> None:
    payload = {
        "executed_vx_body_mps": [0.1, 0.2, 0.3, 0.4, 0.5],
        "executed_vy_body_mps": [0.0, 0.0, 0.1, 0.1, 0.2],
        "executed_yaw_rate_radps": [-0.1, -0.2, -0.3, -0.4, -0.5],
    }

    block = encode_executed_command_block(payload)

    assert block.shape == (ACTIVE_BLOCK_DIM,)
    np.testing.assert_allclose(block[:5], payload["executed_vx_body_mps"])
    np.testing.assert_allclose(block[5:10], payload["executed_vy_body_mps"])
    np.testing.assert_allclose(block[10:15], payload["executed_yaw_rate_radps"])


def test_active_block_to_matrix_returns_per_tick_rows() -> None:
    block = np.array(
        [1, 2, 3, 4, 5, 10, 20, 30, 40, 50, 100, 200, 300, 400, 500],
        dtype=np.float32,
    )

    np.testing.assert_array_equal(
        active_block_to_matrix(block),
        np.array(
            [
                [1, 10, 100],
                [2, 20, 200],
                [3, 30, 300],
                [4, 40, 400],
                [5, 50, 500],
            ],
            dtype=np.float32,
        ),
    )


def test_metadata_compatibility_check() -> None:
    metadata = active_block_metadata()

    assert metadata["active_block_order"] == ACTIVE_BLOCK_ORDER
    assert_active_block_metadata_compatible(metadata)

    bad = dict(metadata)
    bad["active_block_order"] = "row_major_vx_vy_yaw"
    with pytest.raises(ValueError, match="active_block_order"):
        assert_active_block_metadata_compatible(bad)
