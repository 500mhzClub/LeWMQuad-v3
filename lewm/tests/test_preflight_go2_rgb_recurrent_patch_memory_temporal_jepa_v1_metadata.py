from __future__ import annotations

from dataclasses import asdict, replace
import hashlib
import json

import pytest

from lewm.benchmarks import (
    go2_rgb_recurrent_patch_memory_temporal_jepa_v1 as metrics,
)
from scripts import (
    preflight_go2_rgb_recurrent_patch_memory_temporal_jepa_v1_metadata as preflight,
)


def _row(
    *,
    role: str,
    family: str,
    family_index: int,
    scene_index: int,
    row_index: int,
    hold: bool,
) -> dict[str, object]:
    role_offset = 8 if role == "val" else 0
    scene = f"{family}_{role_offset + family_index:02x}{scene_index:010x}"
    environment = row_index % preflight.ENVIRONMENT_COUNT
    base = environment + (row_index % 120) * preflight.ENVIRONMENT_COUNT
    actions = [
        (row_index + position) % metrics.ACTION_COUNT for position in range(6)
    ]
    if hold:
        actions[2] = metrics.HOLD_ACTION_ID
    return {
        "schema": preflight.ROW_SCHEMA,
        "role": role,
        "family": family,
        "scene_id": scene,
        "rgb": [
            (
                f"{scene}/rgb/"
                f"frame_{base + step * preflight.CAUSAL_FRAME_DELTA:06d}_"
                f"env_{environment:02d}.png"
            )
            for step in range(7)
        ],
        "actions": actions,
    }


def _raw(rows: list[dict[str, object]]) -> bytes:
    return b"".join(
        json.dumps(
            row,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
        + b"\n"
        for row in rows
    )


def _fixture() -> tuple[bytes, bytes, preflight.PreflightPolicy]:
    train_values = []
    validation_values = []
    for family_index, family in enumerate(metrics.REGISTERED_FAMILIES):
        for row_offset in range(4):
            train_values.append(
                _row(
                    role="train",
                    family=family,
                    family_index=family_index,
                    scene_index=row_offset % 2,
                    row_index=len(train_values),
                    hold=False,
                )
            )
            validation_values.append(
                _row(
                    role="val",
                    family=family,
                    family_index=family_index,
                    scene_index=row_offset % 2,
                    row_index=len(validation_values),
                    hold=row_offset == 0,
                )
            )
    train_raw = _raw(train_values)
    validation_raw = _raw(validation_values)
    train_rows = preflight.decode_index_bytes(
        train_raw, role="train", expected_rows=len(train_values)
    )
    validation_rows = preflight.decode_index_bytes(
        validation_raw, role="val", expected_rows=len(validation_values)
    )
    schedule = metrics.build_training_schedule(train_rows, rows_per_family=3)
    sentinel = metrics.build_sentinel_indices(
        validation_rows, rows_per_family=3
    )
    full_donors = metrics.build_wrong_history_donor_indices(validation_rows)
    sentinel_donors = metrics.build_wrong_history_donor_indices(
        validation_rows, selected_indices=sentinel
    )
    full_wrong_action = metrics.wrong_action_eligible_indices(validation_rows)
    sentinel_wrong_action = metrics.wrong_action_eligible_indices(
        validation_rows, selected_indices=sentinel
    )
    full_rows, full_scenes = metrics.family_row_scene_counts(
        validation_rows, full_wrong_action
    )
    sentinel_rows, sentinel_scenes = metrics.family_row_scene_counts(
        validation_rows, sentinel_wrong_action
    )
    policy = preflight.PreflightPolicy(
        train_binding=preflight.IndexBinding(
            role="train",
            path="synthetic/train.jsonl",
            file_sha256=hashlib.sha256(train_raw).hexdigest(),
            byte_count=len(train_raw),
            row_count=len(train_rows),
        ),
        validation_binding=preflight.IndexBinding(
            role="val",
            path="synthetic/val.jsonl",
            file_sha256=hashlib.sha256(validation_raw).hexdigest(),
            byte_count=len(validation_raw),
            row_count=len(validation_rows),
        ),
        output_root="synthetic/metadata_preflight/attempt_v1",
        train_rows_per_family=3,
        sentinel_rows_per_family=3,
        train_schedule_sha256=metrics.canonical_json_sha256(schedule),
        sentinel_indices_sha256=metrics.canonical_json_sha256(sentinel),
        full_donor_sha256=metrics.canonical_json_sha256(full_donors),
        sentinel_donor_sha256=metrics.canonical_json_sha256(sentinel_donors),
        train_scene_counts={family: 2 for family in metrics.REGISTERED_FAMILIES},
        validation_scene_counts={
            family: 2 for family in metrics.REGISTERED_FAMILIES
        },
        sentinel_scene_count=len(
            {validation_rows[index].scene_id for index in sentinel}
        ),
        full_wrong_action_row_counts=full_rows,
        full_wrong_action_scene_counts=full_scenes,
        sentinel_wrong_action_row_counts=sentinel_rows,
        sentinel_wrong_action_scene_counts=sentinel_scenes,
        require_all_visible_actions=False,
    )
    return train_raw, validation_raw, policy


def _rebind(
    policy: preflight.PreflightPolicy,
    *,
    train_raw: bytes | None = None,
    validation_raw: bytes | None = None,
) -> preflight.PreflightPolicy:
    result = policy
    if train_raw is not None:
        result = replace(
            result,
            train_binding=replace(
                result.train_binding,
                file_sha256=hashlib.sha256(train_raw).hexdigest(),
                byte_count=len(train_raw),
            ),
        )
    if validation_raw is not None:
        result = replace(
            result,
            validation_binding=replace(
                result.validation_binding,
                file_sha256=hashlib.sha256(validation_raw).hexdigest(),
                byte_count=len(validation_raw),
            ),
        )
    return result


def _authority(
    policy: preflight.PreflightPolicy,
    *,
    repository_root: str = "/synthetic/repository",
    output_root: str = "synthetic/metadata_preflight/attempt_v1",
) -> tuple[preflight.ValidatedAuthority, dict[str, object]]:
    core = {
        "schema": preflight.AUTHORITY_SCHEMA,
        "status": preflight.AUTHORITY_STATUS,
        "preregistration_commit": metrics.PREREGISTRATION_COMMIT,
        "one_shot": True,
        "repository_root": repository_root,
        "output_root": output_root,
        "output_root_absent_at_authorization": True,
        "train_index": asdict(policy.train_binding),
        "validation_index": asdict(policy.validation_binding),
    }
    core["content_sha256"] = hashlib.sha256(
        json.dumps(
            core,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
    ).hexdigest()
    raw = (
        json.dumps(
            core,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
        + b"\n"
    )
    authority = preflight.validate_authority_bytes(
        raw,
        authority_path="docs/synthetic_temporal_metadata_authority.json",
        policy=policy,
    )
    reservation = preflight.build_reservation(
        authority, created_utc="2026-07-31T12:00:00Z"
    )
    return authority, reservation


def test_synthetic_preflight_passes_without_opening_any_referenced_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    train_raw, validation_raw, policy = _fixture()
    authority, reservation = _authority(policy)

    def forbidden_open(*_args: object, **_kwargs: object) -> int:
        raise AssertionError("byte-level preflight attempted filesystem access")

    monkeypatch.setattr(preflight.os, "open", forbidden_open)
    receipt = preflight.preflight_from_bytes(
        train_raw=train_raw,
        validation_raw=validation_raw,
        authority=authority,
        reservation=reservation,
        policy=policy,
    )
    assert receipt["status"] == "PASS_METADATA_PREFLIGHT"
    assert all(receipt["checks"].values())
    assert receipt["authority"] == dict(authority.binding)
    assert receipt["access"]["metadata_index_open_count"] == 0
    assert receipt["access"]["metadata_row_count"] == 64
    assert receipt["access"]["rgb_path_string_count"] == 448
    assert receipt["access"]["action_id_count"] == 384
    assert receipt["access"]["rgb_open_count"] == 0
    assert receipt["access"]["checkpoint_open_count"] == 0
    assert receipt["access"]["navigation_open_count"] == 0
    assert receipt["access"]["held_out_or_sealed_opened"] is False
    core = dict(receipt)
    content_sha256 = core.pop("content_sha256")
    assert hashlib.sha256(
        json.dumps(
            core,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
    ).hexdigest() == content_sha256


def test_decoder_rejects_duplicate_keys_and_noncanonical_json() -> None:
    family = metrics.REGISTERED_FAMILIES[0]
    value = _row(
        role="train",
        family=family,
        family_index=0,
        scene_index=0,
        row_index=0,
        hold=False,
    )
    canonical = _raw([value])
    duplicate = canonical[:-2] + b',\"role\":\"train\"}\n'
    with pytest.raises(preflight.MetadataPreflightError, match="duplicate"):
        preflight.decode_index_bytes(
            duplicate, role="train", expected_rows=1
        )

    noncanonical = json.dumps(value).encode("utf-8") + b"\n"
    with pytest.raises(preflight.MetadataPreflightError, match="canonical"):
        preflight.decode_index_bytes(
            noncanonical, role="train", expected_rows=1
        )


def test_authority_is_content_bound_and_attempt_root_is_one_shot(
    tmp_path,
) -> None:
    _, _, policy = _fixture()
    with pytest.raises(preflight.MetadataPreflightError, match="scope"):
        _authority(policy, output_root="arbitrary/attempt_v99")
    parent = tmp_path / "synthetic" / "metadata_preflight"
    parent.mkdir(parents=True)
    authority, _ = _authority(
        policy,
        repository_root=tmp_path.resolve().as_posix(),
    )
    output, reservation = preflight.reserve_attempt_root(
        tmp_path,
        authority,
        created_utc="2026-07-31T12:00:00Z",
    )
    assert output == parent / "attempt_v1"
    assert (output / "reservation.json").is_file()
    assert reservation["authority"] == dict(authority.binding)
    with pytest.raises(preflight.MetadataPreflightError, match="not absent"):
        preflight.reserve_attempt_root(
            tmp_path,
            authority,
            created_utc="2026-07-31T12:00:01Z",
        )

    malformed = dict(authority.value)
    malformed["content_sha256"] = "0" * 64
    malformed_raw = (
        json.dumps(
            malformed,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("ascii")
        + b"\n"
    )
    with pytest.raises(preflight.MetadataPreflightError, match="content binding"):
        preflight.validate_authority_bytes(
            malformed_raw,
            authority_path=authority.binding["path"],
            policy=policy,
        )


def test_preflight_rejects_binding_schedule_and_hold_identity_changes() -> None:
    train_raw, validation_raw, policy = _fixture()
    authority, reservation = _authority(policy)
    with pytest.raises(preflight.MetadataPreflightError, match="binding"):
        preflight.preflight_from_bytes(
            train_raw=train_raw + b"\n",
            validation_raw=validation_raw,
            authority=authority,
            reservation=reservation,
            policy=policy,
        )

    changed_policy = replace(policy, train_schedule_sha256="0" * 64)
    changed_authority, changed_reservation = _authority(changed_policy)
    with pytest.raises(preflight.MetadataPreflightError, match="schedule"):
        preflight.preflight_from_bytes(
            train_raw=train_raw,
            validation_raw=validation_raw,
            authority=changed_authority,
            reservation=changed_reservation,
            policy=changed_policy,
        )

    rows = [
        json.loads(line)
        for line in validation_raw.decode("ascii").splitlines()
    ]
    rows[1]["actions"][2] = metrics.HOLD_ACTION_ID
    changed = _raw(rows)
    rebound = _rebind(policy, validation_raw=changed)
    rebound_authority, rebound_reservation = _authority(rebound)
    with pytest.raises(
        preflight.MetadataPreflightError,
        match="wrong-action eligibility",
    ):
        preflight.preflight_from_bytes(
            train_raw=train_raw,
            validation_raw=changed,
            authority=rebound_authority,
            reservation=rebound_reservation,
            policy=rebound,
        )


def test_preflight_rejects_train_validation_scene_or_rgb_overlap() -> None:
    train_raw, validation_raw, policy = _fixture()
    train_first = json.loads(train_raw.decode("ascii").splitlines()[0])
    rows = [
        json.loads(line)
        for line in validation_raw.decode("ascii").splitlines()
    ]
    old_scene = rows[0]["scene_id"]
    for row in rows:
        if row["scene_id"] == old_scene:
            row["scene_id"] = train_first["scene_id"]
            row["rgb"] = [
                leaf.replace(old_scene, train_first["scene_id"])
                for leaf in row["rgb"]
            ]
    assert old_scene != rows[0]["scene_id"]
    changed = _raw(rows)
    rebound = _rebind(policy, validation_raw=changed)
    authority, reservation = _authority(rebound)
    with pytest.raises(
        preflight.MetadataPreflightError,
        match="scenes overlap",
    ):
        preflight.preflight_from_bytes(
            train_raw=train_raw,
            validation_raw=changed,
            authority=authority,
            reservation=reservation,
            policy=rebound,
        )
