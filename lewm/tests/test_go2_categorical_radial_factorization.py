from __future__ import annotations

import numpy as np
import pytest
import torch

from lewm.benchmarks.go2_categorical_radial_factorization import (
    POLAR_SHAPE,
    audit_exact_cartesian_roundtrip,
    audit_mapping_injectivity,
    build_cartesian_to_polar_bin_mapping,
    build_radial_factorization,
    force_outside_fov_unknown,
    gather_polar_logits_to_cartesian,
    gather_radial_labels_to_cartesian,
    geometry_metadata,
    representable_cartesian_mask,
    scatter_cartesian_labels_to_radial,
)
from lewm.datasets.go2_paired_navigation import (
    FREE_CLASS,
    OCCUPIED_CLASS,
    UNKNOWN_CLASS,
)


def test_fixed_v1_geometry_is_injective() -> None:
    metadata = geometry_metadata()
    mapping = build_cartesian_to_polar_bin_mapping()
    audit = audit_mapping_injectivity(mapping)

    assert metadata["cartesian_shape"] == [64, 64]
    assert metadata["horizontal_fov_deg"] == 78.323
    assert metadata["half_fov_deg"] == pytest.approx(78.323 / 2.0)
    assert metadata["radial_bin_count"] == 64
    assert metadata["radial_bin_size_m"] == 0.1
    assert metadata["radial_range_m"] == [0.0, 6.4]
    assert metadata["angular_bin_count"] == 256
    assert metadata["polar_shape"] == [64, 256]
    assert audit["representable_cartesian_cell_count"] == 2198
    assert audit["mapped_cartesian_cell_count"] == 2198
    assert audit["unique_polar_bin_count"] == 2198
    assert audit["collision_count"] == 0
    assert audit["injective"] is True


def test_mapping_audit_rejects_a_polar_bin_collision() -> None:
    mapping = build_cartesian_to_polar_bin_mapping()
    supported_cells = np.argwhere(np.all(mapping >= 0, axis=-1))
    first = tuple(supported_cells[0])
    second = tuple(supported_cells[1])
    mapping[second] = mapping[first]

    with pytest.raises(ValueError, match="mapping collision"):
        audit_mapping_injectivity(mapping)

    labels = np.full((64, 64), UNKNOWN_CLASS, dtype=np.uint8)
    with pytest.raises(ValueError, match="mapping collision"):
        scatter_cartesian_labels_to_radial(labels, mapping=mapping)


@pytest.mark.parametrize("mutation", ["axis", "angle_sign", "range_bin"])
def test_mapping_audit_rejects_geometry_mutations(mutation: str) -> None:
    mapping = build_cartesian_to_polar_bin_mapping()
    supported = np.all(mapping >= 0, axis=-1)
    if mutation == "axis":
        mapping = mapping.transpose(1, 0, 2).copy()
    elif mutation == "angle_sign":
        mapping[..., 1][supported] = 255 - mapping[..., 1][supported]
    else:
        mapping[..., 0][supported] = 63 - mapping[..., 0][supported]

    with pytest.raises(
        ValueError,
        match="fixed front-camera support|deterministic v1",
    ):
        audit_mapping_injectivity(mapping)


def test_known_label_outside_fov_is_rejected_or_explicitly_forced_unknown() -> None:
    mapping = build_cartesian_to_polar_bin_mapping()
    support = representable_cartesian_mask(mapping)
    outside = tuple(np.argwhere(~support)[0])
    labels = np.full((64, 64), UNKNOWN_CLASS, dtype=np.uint8)
    labels[outside] = FREE_CLASS

    with pytest.raises(ValueError, match="known Cartesian labels outside"):
        scatter_cartesian_labels_to_radial(labels, mapping=mapping)
    with pytest.raises(ValueError, match="all known Cartesian labels"):
        audit_exact_cartesian_roundtrip(labels, mapping=mapping)

    forced = force_outside_fov_unknown(labels, mapping=mapping)
    assert forced[outside] == UNKNOWN_CLASS
    radial = scatter_cartesian_labels_to_radial(
        labels, mapping=mapping, reject_outside_known=False
    )
    gathered = gather_radial_labels_to_cartesian(radial, mapping=mapping)
    assert gathered[outside] == UNKNOWN_CLASS


def test_random_representable_integer_labels_roundtrip_exactly() -> None:
    mapping = build_cartesian_to_polar_bin_mapping()
    support = representable_cartesian_mask(mapping)
    rng = np.random.default_rng(20260710)
    labels = rng.integers(0, 3, size=(64, 64), dtype=np.uint8)
    labels[~support] = UNKNOWN_CLASS

    radial = scatter_cartesian_labels_to_radial(labels, mapping=mapping)
    recovered = gather_radial_labels_to_cartesian(radial, mapping=mapping)
    audit = audit_exact_cartesian_roundtrip(labels, mapping=mapping)

    assert radial.shape == POLAR_SHAPE
    assert radial.dtype == labels.dtype
    np.testing.assert_array_equal(recovered, labels)
    assert np.all(recovered[~support] == UNKNOWN_CLASS)
    assert audit["exact_roundtrip"] is True
    assert audit["roundtrip_mismatch_count"] == 0
    assert audit["known_cartesian_cell_count"] == int(
        np.count_nonzero(labels != UNKNOWN_CLASS)
    )


def test_categorical_transforms_reject_non_integer_or_unknown_classes() -> None:
    float_labels = np.zeros((64, 64), dtype=np.float32)
    with pytest.raises(ValueError, match="integer dtype"):
        scatter_cartesian_labels_to_radial(float_labels)

    invalid = np.full((64, 64), UNKNOWN_CLASS, dtype=np.int64)
    invalid[32, 32] = 7
    with pytest.raises(ValueError, match="invalid classes"):
        force_outside_fov_unknown(invalid)

    radial = np.full(POLAR_SHAPE, UNKNOWN_CLASS, dtype=np.int8)
    radial[0, 0] = OCCUPIED_CLASS + 1
    with pytest.raises(ValueError, match="invalid classes"):
        gather_radial_labels_to_cartesian(radial)


def test_model_factorization_exposes_flat_indices_and_bin_centers() -> None:
    factorization = build_radial_factorization()
    flat = factorization.cartesian_to_polar_flat_indices

    assert flat.shape == (4096,)
    assert flat.dtype == np.int64
    assert factorization.representable_mask.shape == (64, 64)
    assert np.array_equal(
        factorization.representable_mask.reshape(-1), flat >= 0
    )
    assert np.unique(flat[flat >= 0]).size == 2198
    assert factorization.radial_centers_m.shape == (64,)
    assert factorization.radial_centers_m[[0, -1]].tolist() == pytest.approx(
        [0.05, 6.35]
    )
    assert factorization.angular_centers_rad.shape == (256,)
    assert factorization.angular_centers_rad[0] < 0.0
    assert factorization.angular_centers_rad[-1] > 0.0
    assert factorization.angular_centers_rad[0] == pytest.approx(
        -factorization.angular_centers_rad[-1]
    )
    assert not flat.flags.writeable
    assert not factorization.representable_mask.flags.writeable


def test_polar_logits_gather_is_exact_finite_and_forces_unknown_outside() -> None:
    factorization = build_radial_factorization()
    polar_logits = torch.arange(
        2 * 3 * 64 * 256, dtype=torch.float32
    ).reshape(2, 3, 64, 256)
    polar_logits.requires_grad_(True)

    cartesian = gather_polar_logits_to_cartesian(
        polar_logits, factorization, unknown_logit=7.0
    )

    assert cartesian.shape == (2, 3, 64, 64)
    assert torch.isfinite(cartesian).all()
    flat_cartesian = cartesian.flatten(start_dim=-2)
    supported = factorization.cartesian_to_polar_flat_indices >= 0
    indices = torch.from_numpy(
        factorization.cartesian_to_polar_flat_indices[supported].copy()
    )
    torch.testing.assert_close(
        flat_cartesian[..., supported],
        polar_logits.flatten(start_dim=-2).index_select(-1, indices),
    )
    assert torch.all(flat_cartesian[..., UNKNOWN_CLASS, ~supported] == 7.0)
    assert torch.all(
        flat_cartesian[..., FREE_CLASS, ~supported]
        == torch.finfo(torch.float32).min
    )
    assert torch.all(
        flat_cartesian[..., OCCUPIED_CLASS, ~supported]
        == torch.finfo(torch.float32).min
    )
    assert torch.all(
        flat_cartesian[..., ~supported].argmax(dim=-2) == UNKNOWN_CLASS
    )

    cartesian.sum().backward()
    assert polar_logits.grad is not None
    assert torch.isfinite(polar_logits.grad).all()
