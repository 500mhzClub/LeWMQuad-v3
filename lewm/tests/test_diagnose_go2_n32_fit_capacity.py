from __future__ import annotations

import numpy as np
import pytest

from scripts import diagnose_go2_n32_fit_capacity as diagnostic


def test_perfect_logits_recover_every_label() -> None:
    labels = np.arange(2 * 64 * 64, dtype=np.int64).reshape(2, 64, 64) % 3
    logits = diagnostic.perfect_logits(labels, 12.0)
    assert logits.shape == (2, 3, 64, 64)
    assert np.array_equal(logits.argmax(axis=1), labels)
    assert np.isfinite(logits).all()


def test_control_source_indices_are_exact_and_fail_closed() -> None:
    records = [
        {"image_sha256": "a", "control_image_sha256": "b"},
        {"image_sha256": "b", "control_image_sha256": "a"},
    ]
    assert diagnostic._control_source_indices(
        records, "control_image_sha256"
    ).tolist() == [1, 0]
    records[0]["control_image_sha256"] = "outside"
    with pytest.raises(ValueError, match="outside"):
        diagnostic._control_source_indices(records, "control_image_sha256")
