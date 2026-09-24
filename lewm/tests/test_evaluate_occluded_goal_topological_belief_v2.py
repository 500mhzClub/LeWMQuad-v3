from __future__ import annotations

import copy
import hashlib
import inspect
import json
import os
from pathlib import Path
import struct
import subprocess
import sys
import zipfile

import pytest

from lewm.safety import occluded_goal_topological_belief_metrics_v2 as METRICS
from scripts import evaluate_occluded_goal_topological_belief_v2 as E


def _attach(value: dict[str, object]) -> dict[str, object]:
    return {**value, "content_digest": E.canonical_digest(value)}


def _npy(descr: str, shape: tuple[int, ...], payload: bytes) -> bytes:
    header = repr(
        {"descr": descr, "fortran_order": False, "shape": shape}
    ).encode("latin1")
    padding = (16 - ((10 + len(header) + 1) % 16)) % 16
    header = header + (b" " * padding) + b"\n"
    return b"\x93NUMPY\x01\x00" + struct.pack("<H", len(header)) + header + payload


def _string_npy(values: list[str]) -> bytes:
    width = max(len(value.encode("ascii")) for value in values)
    payload = b"".join(value.encode("ascii").ljust(width, b"\x00") for value in values)
    return _npy(f"|S{width}", (len(values),), payload)


def _write_npz(path: Path, members: dict[str, bytes]) -> None:
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, raw in members.items():
            archive.writestr(f"{name}.npy", raw)


def _binding(path: Path) -> dict[str, object]:
    raw = path.read_bytes()
    return {"path": path.name, "bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}


def test_locked_metrics_authorities_match_reducer_contract() -> None:
    authority = E._validate_authority(METRICS)
    cache = METRICS.cache_gate_authority()
    assert authority["cache_gate"] == cache
    assert authority["v1_retained_root"] == METRICS.v1_retained_root_authority()
    assert cache["receipt_self_digests"] is False
    assert set(inspect.signature(METRICS.validate_cache_gate).parameters) == {
        "pixel_index",
        "template_index",
        "occurrence_index",
        "canonical_latent_index",
        "canonical_descriptor_index",
        "encoding_receipt",
        "cache_receipt",
    }


def test_system_python_cli_and_import_need_no_pythonpath() -> None:
    environment = os.environ.copy()
    environment.pop("PYTHONPATH", None)
    help_result = subprocess.run(
        [sys.executable, str(E.ROOT / E.REDUCER_SOURCE_PATH), "--help"],
        cwd=E.ROOT,
        env=environment,
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    assert help_result.returncode == 0, help_result.stderr
    assert "--output-root" in help_result.stdout
    import_result = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "from scripts import "
                "evaluate_occluded_goal_topological_belief_v2 as e; "
                "assert e.EXPERIMENT_ID == 'OCCLUDED_GOAL_TOPOLOGICAL_BELIEF_V2'"
            ),
        ],
        cwd=E.ROOT,
        env=environment,
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    assert import_result.returncode == 0, import_result.stderr


def test_runtime_contract_binds_freeze_and_complete_v1_authority() -> None:
    v1 = METRICS.v1_retained_root_authority()
    scientific = _attach(
        {
            "schema": "occluded_goal_topological_belief_v2.contract.v1",
            "experiment_id": E.EXPERIMENT_ID,
            "v1_retained_root_binding": v1,
        }
    )
    runtime = _attach(
        {
            "schema": "occluded_goal_topological_belief_v2.runtime_contract.v1",
            "source_freeze_commit": "a" * 40,
            "parent_commit": "b" * 40,
            "scientific_contract": scientific,
            "v1_binding_verified_before_copy": v1,
        }
    )
    E._validate_runtime_contract_binding(
        runtime, source_freeze_commit="a" * 40, v1_authority=v1
    )
    altered = copy.deepcopy(runtime)
    altered.pop("content_digest")
    altered["v1_binding_verified_before_copy"] = {"root": "wrong"}
    altered = _attach(altered)
    with pytest.raises(E.RegenerationError, match="pre-copy V1 verification"):
        E._validate_runtime_contract_binding(
            altered, source_freeze_commit="a" * 40, v1_authority=v1
        )


def test_npz_parser_recomputes_each_canonical_row_and_rejects_schema_drift(
    tmp_path: Path,
) -> None:
    identities = ["1" * 64, "2" * 64]
    first_row = bytes(range(8))
    second_row = bytes(range(8, 16))
    path = tmp_path / "canonical_tokens.npz"

    def write(payload: bytes, schema: str = "synthetic.tokens.v1") -> None:
        _write_npz(
            path,
            {
                "schema": _string_npy([schema]),
                "pixel_sha256": _string_npy(identities),
                "raw_tokens": _npy("<f2", (2, 2, 2), payload),
            },
        )

    write(first_row + second_row)
    root_fd = os.open(tmp_path, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        _, strings, row_hashes, raw_hashes = E._inspect_npz_at(
            root_fd,
            path.name,
            expected_members={"schema", "pixel_sha256", "raw_tokens"},
            expected_schema="synthetic.tokens.v1",
            identity_member="pixel_sha256",
            payload_member="raw_tokens",
            payload_shape=(2, 2, 2),
            payload_descr="<f2",
            payload_dtype="<f2",
            payload_item_size=2,
        )
        assert strings["pixel_sha256"] == identities
        assert row_hashes == [
            E._array_row_digest(first_row, (2, 2), "<f2"),
            E._array_row_digest(second_row, (2, 2), "<f2"),
        ]
        assert raw_hashes == [
            hashlib.sha256(first_row).hexdigest(),
            hashlib.sha256(second_row).hexdigest(),
        ]

        tampered = bytearray(second_row)
        tampered[-1] ^= 1
        write(first_row + bytes(tampered))
        _, _, changed_hashes, _ = E._inspect_npz_at(
            root_fd,
            path.name,
            expected_members={"schema", "pixel_sha256", "raw_tokens"},
            expected_schema="synthetic.tokens.v1",
            identity_member="pixel_sha256",
            payload_member="raw_tokens",
            payload_shape=(2, 2, 2),
            payload_descr="<f2",
            payload_dtype="<f2",
            payload_item_size=2,
        )
        assert changed_hashes[0] == row_hashes[0]
        assert changed_hashes[1] != row_hashes[1]

        write(first_row + second_row, schema="wrong.tokens.v1")
        with pytest.raises(E.RegenerationError, match="schema string drift"):
            E._inspect_npz_at(
                root_fd,
                path.name,
                expected_members={"schema", "pixel_sha256", "raw_tokens"},
                expected_schema="synthetic.tokens.v1",
                identity_member="pixel_sha256",
                payload_member="raw_tokens",
                payload_shape=(2, 2, 2),
                payload_descr="<f2",
                payload_dtype="<f2",
                payload_item_size=2,
            )
    finally:
        os.close(root_fd)


def test_canonical_index_rejects_single_row_payload_tamper_and_slot_metadata(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(E, "EXPECTED_UNIQUE_PIXEL_COUNT", 2)
    pixel_ids = ["1" * 64, "2" * 64]
    pixel_rows = [
        {
            "pixel_index": index,
            "pixel_sha256": pixel_sha,
            "canonical_template_id": f"template-{index}",
        }
        for index, pixel_sha in enumerate(pixel_ids)
    ]
    original_token_rows = [bytes(range(8)), bytes(range(8, 16))]
    changed_second = bytearray(original_token_rows[1])
    changed_second[0] ^= 1
    descriptor_rows = [bytes(range(16)), bytes(range(16, 32))]
    token_path = tmp_path / "canonical_tokens.npz"
    descriptor_path = tmp_path / "canonical_descriptors.npz"
    _write_npz(
        token_path,
        {
            "schema": _string_npy(["synthetic.tokens.v1"]),
            "pixel_sha256": _string_npy(pixel_ids),
            "raw_tokens": _npy(
                "<f2", (2, 2, 2), original_token_rows[0] + bytes(changed_second)
            ),
        },
    )
    _write_npz(
        descriptor_path,
        {
            "schema": _string_npy(["synthetic.descriptors.v1"]),
            "pixel_sha256": _string_npy(pixel_ids),
            "spatial_descriptors": _npy(
                "<f4", (2, 2, 2), b"".join(descriptor_rows)
            ),
        },
    )
    root_fd = os.open(tmp_path, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        token_binding, token_strings, actual_token_hashes, _ = E._inspect_npz_at(
            root_fd,
            token_path.name,
            expected_members={"schema", "pixel_sha256", "raw_tokens"},
            expected_schema="synthetic.tokens.v1",
            identity_member="pixel_sha256",
            payload_member="raw_tokens",
            payload_shape=(2, 2, 2),
            payload_descr="<f2",
            payload_dtype="<f2",
            payload_item_size=2,
        )
        descriptor_binding, descriptor_strings, descriptor_hashes, _ = E._inspect_npz_at(
            root_fd,
            descriptor_path.name,
            expected_members={"schema", "pixel_sha256", "spatial_descriptors"},
            expected_schema="synthetic.descriptors.v1",
            identity_member="pixel_sha256",
            payload_member="spatial_descriptors",
            payload_shape=(2, 2, 2),
            payload_descr="<f4",
            payload_dtype="<f4",
            payload_item_size=4,
        )
    finally:
        os.close(root_fd)
    stale_token_hashes = [
        E._array_row_digest(row, (2, 2), "<f2") for row in original_token_rows
    ]
    receipt_binding = {
        "path": "encoding_determinism_receipt.json",
        "bytes": 11,
        "sha256": "e" * 64,
    }
    pixel_digest = "f" * 64

    def indices(token_hashes: list[str]) -> tuple[dict[str, object], dict[str, object]]:
        latent = _attach(
            {
                "schema": "synthetic.latent_index.v1",
                "experiment_id": E.EXPERIMENT_ID,
                "source_freeze_commit": "a" * 40,
                "pixel_index_binding": pixel_digest,
                "encoding_determinism_receipt_binding": receipt_binding,
                "tokens_file": token_binding,
                "records": [
                    {
                        "pixel_index": index,
                        "pixel_sha256": pixel_ids[index],
                        "canonical_template_id": f"template-{index}",
                        "raw_token_row_index": index,
                        "raw_token_sha256": token_hashes[index],
                    }
                    for index in range(2)
                ],
            }
        )
        descriptor = _attach(
            {
                "schema": "synthetic.descriptor_index.v1",
                "experiment_id": E.EXPERIMENT_ID,
                "source_freeze_commit": "a" * 40,
                "pixel_index_binding": pixel_digest,
                "encoding_determinism_receipt_binding": receipt_binding,
                "descriptors_file": descriptor_binding,
                "records": [
                    {
                        "pixel_index": index,
                        "pixel_sha256": pixel_ids[index],
                        "canonical_template_id": f"template-{index}",
                        "spatial_descriptor_row_index": index,
                        "spatial_descriptor_sha256": descriptor_hashes[index],
                    }
                    for index in range(2)
                ],
            }
        )
        return latent, descriptor

    stale_latent, descriptor = indices(stale_token_hashes)
    with pytest.raises(E.RegenerationError, match="actual canonical-array bytes"):
        E._validate_canonical_indices(
            stale_latent,
            descriptor,
            pixel_rows=pixel_rows,
            token_binding=token_binding,
            descriptor_binding=descriptor_binding,
            token_pixel_ids=token_strings["pixel_sha256"],
            descriptor_pixel_ids=descriptor_strings["pixel_sha256"],
            token_row_hashes=actual_token_hashes,
            descriptor_row_hashes=descriptor_hashes,
            pixel_index_content_digest=pixel_digest,
            encoding_receipt_binding=receipt_binding,
        )

    latent, descriptor = indices(actual_token_hashes)
    E._validate_canonical_indices(
        latent,
        descriptor,
        pixel_rows=pixel_rows,
        token_binding=token_binding,
        descriptor_binding=descriptor_binding,
        token_pixel_ids=token_strings["pixel_sha256"],
        descriptor_pixel_ids=descriptor_strings["pixel_sha256"],
        token_row_hashes=actual_token_hashes,
        descriptor_row_hashes=descriptor_hashes,
        pixel_index_content_digest=pixel_digest,
        encoding_receipt_binding=receipt_binding,
    )
    latent_with_slot = copy.deepcopy(latent)
    latent_with_slot.pop("content_digest")
    latent_with_slot["records"][0]["batch_slot"] = 0
    latent_with_slot = _attach(latent_with_slot)
    with pytest.raises(E.RegenerationError, match="row must be an object"):
        E._validate_canonical_indices(
            latent_with_slot,
            descriptor,
            pixel_rows=pixel_rows,
            token_binding=token_binding,
            descriptor_binding=descriptor_binding,
            token_pixel_ids=token_strings["pixel_sha256"],
            descriptor_pixel_ids=descriptor_strings["pixel_sha256"],
            token_row_hashes=actual_token_hashes,
            descriptor_row_hashes=descriptor_hashes,
            pixel_index_content_digest=pixel_digest,
            encoding_receipt_binding=receipt_binding,
        )


def test_observation_bytes_are_the_authority_for_pixel_and_occurrence_fanout(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(E, "EXPECTED_TEMPLATE_COUNT", 3)
    monkeypatch.setattr(E, "EXPECTED_UNIQUE_PIXEL_COUNT", 2)
    monkeypatch.setattr(E, "EXPECTED_REUSE_COUNT", 1)
    monkeypatch.setattr(E, "EXPECTED_OCCURRENCE_COUNT", 3)
    image_size = 168 * 224 * 3
    images = [bytes(image_size), bytes(image_size), bytes(image_size - 1) + b"\x01"]
    template_ids = ["template-0", "template-1", "template-2"]
    raw_hashes = [hashlib.sha256(row).hexdigest() for row in images]
    observations = tmp_path / "observations.npz"
    _write_npz(
        observations,
        {
            "schema": _string_npy(
                ["occluded_goal_topological_belief_v1.observations.v1"]
            ),
            "pixel_template_ids": _string_npy(template_ids),
            "image_sha256": _string_npy(raw_hashes),
            "images": _npy("|u1", (3, 168, 224, 3), b"".join(images)),
        },
    )
    captures = [
        {
            "capture_id": f"capture-{index}",
            "episode_id": "episode-0",
            "node_id": f"node-{index}",
            "phase": "PHASE_A_TRAVERSAL",
            "timestamp_s": float(index),
            "pixel_template_id": template_ids[index],
            "query_ids": [],
        }
        for index in range(3)
    ]
    keyframe = _attach(
        {
            "schema": "occluded_goal_topological_belief_v1.keyframe_index.v1",
            "observations_file": _binding(observations),
            "image_shape": [168, 224, 3],
            "unique_pixel_template_count": 3,
            "capture_count": 3,
            "phase_a_capture_count": 3,
            "phase_c_capture_count": 0,
            "query_reference_disjoint": True,
            "records": [
                {
                    "pixel_template_id": template_id,
                    "row_index": index,
                    "image_sha256": raw_hashes[index],
                    "recipe": {"synthetic": index},
                }
                for index, template_id in enumerate(template_ids)
            ],
            "captures": captures,
        }
    )
    root_fd = os.open(tmp_path, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        evidence = E._inspect_observations_at(root_fd, keyframe)
    finally:
        os.close(root_fd)
    keyframe_raw = E.canonical_document_bytes(keyframe)
    evidence["keyframe_bytes"] = len(keyframe_raw)
    evidence["keyframe_sha256"] = hashlib.sha256(keyframe_raw).hexdigest()
    groups: dict[str, list[int]] = {}
    for index, pixel_sha in enumerate(evidence["pixel_sha256s"]):
        groups.setdefault(pixel_sha, []).append(index)
    pixel_rows = []
    pixel_by_template: dict[str, dict[str, object]] = {}
    for pixel_index, pixel_sha in enumerate(sorted(groups)):
        indices = groups[pixel_sha]
        members = [template_ids[index] for index in indices]
        row = {
            "pixel_index": pixel_index,
            "pixel_sha256": pixel_sha,
            "canonical_template_id": min(members),
            "member_template_ids": members,
            "observation_row_indices": indices,
        }
        pixel_rows.append(row)
        for member in members:
            pixel_by_template[member] = row
    pixel_index = _attach(
        {
            "schema": "occluded_goal_topological_belief_v2.pixel_index.v1",
            "experiment_id": E.EXPERIMENT_ID,
            "observations_binding": evidence["binding"],
            "identity_rule": METRICS.cache_gate_authority()["hash_domains"][
                "rgb_pixel_sha256"
            ],
            "records": pixel_rows,
        }
    )
    template_rows = [
        {
            "template_row_index": index,
            "pixel_template_id": template_id,
            "pixel_sha256": pixel_by_template[template_id]["pixel_sha256"],
            "pixel_index": pixel_by_template[template_id]["pixel_index"],
            "canonical_template_id": pixel_by_template[template_id][
                "canonical_template_id"
            ],
        }
        for index, template_id in enumerate(template_ids)
    ]
    template_index = _attach(
        {
            "schema": (
                "occluded_goal_topological_belief_v2.template_to_pixel_index.v1"
            ),
            "experiment_id": E.EXPERIMENT_ID,
            "pixel_index_binding": pixel_index["content_digest"],
            "records": template_rows,
        }
    )
    template_by_id = {row["pixel_template_id"]: row for row in template_rows}
    occurrence_index = _attach(
        {
            "schema": "occluded_goal_topological_belief_v2.occurrence_index.v1",
            "experiment_id": E.EXPERIMENT_ID,
            "keyframe_index_binding": {
                "path": "keyframe_index.json",
                "bytes": len(keyframe_raw),
                "sha256": hashlib.sha256(keyframe_raw).hexdigest(),
            },
            "template_index_binding": template_index["content_digest"],
            "records": [
                {
                    "occurrence_index": index,
                    **capture,
                    "pixel_sha256": template_by_id[capture["pixel_template_id"]][
                        "pixel_sha256"
                    ],
                    "pixel_index": template_by_id[capture["pixel_template_id"]][
                        "pixel_index"
                    ],
                }
                for index, capture in enumerate(captures)
            ],
        }
    )
    result = E._validate_pixel_documents(
        pixel_index,
        template_index,
        occurrence_index,
        keyframe,
        evidence,
        rgb_hash_domain=METRICS.cache_gate_authority()["hash_domains"][
            "rgb_pixel_sha256"
        ],
    )
    assert result["unique_pixel_count"] == 2
    altered = copy.deepcopy(pixel_index)
    altered.pop("content_digest")
    altered["records"][0]["gpu_batch_slot"] = 0
    altered = _attach(altered)
    altered_template = copy.deepcopy(template_index)
    altered_template.pop("content_digest")
    altered_template["pixel_index_binding"] = altered["content_digest"]
    altered_template = _attach(altered_template)
    altered_occurrence = copy.deepcopy(occurrence_index)
    altered_occurrence.pop("content_digest")
    altered_occurrence["template_index_binding"] = altered_template["content_digest"]
    altered_occurrence = _attach(altered_occurrence)
    with pytest.raises(E.RegenerationError, match="pixel index row"):
        E._validate_pixel_documents(
            altered,
            altered_template,
            altered_occurrence,
            keyframe,
            evidence,
            rgb_hash_domain=METRICS.cache_gate_authority()["hash_domains"][
                "rgb_pixel_sha256"
            ],
        )


def test_cache_receipts_are_ordinary_documents_and_bind_actual_tensor_rows() -> None:
    pixel_shas = sorted(
        hashlib.sha256(f"pixel-{index}".encode()).hexdigest()
        for index in range(E.EXPECTED_UNIQUE_PIXEL_COUNT)
    )
    pixel_rows = [
        {
            "pixel_index": index,
            "pixel_sha256": pixel_sha,
            "canonical_template_id": f"template-{index:03d}",
        }
        for index, pixel_sha in enumerate(pixel_shas)
    ]
    token_hashes = [hashlib.sha256(f"token-{index}".encode()).hexdigest() for index in range(len(pixel_rows))]
    descriptor_hashes = [hashlib.sha256(f"descriptor-{index}".encode()).hexdigest() for index in range(len(pixel_rows))]
    records = [
        {
            "pixel_index": index,
            "pixel_sha256": pixel["pixel_sha256"],
            "canonical_template_id": pixel["canonical_template_id"],
            "preprocessed_tensor_sha256": hashlib.sha256(
                f"preprocessed-{index}".encode()
            ).hexdigest(),
            "raw_token_sha256": token_hashes[index],
            "spatial_descriptor_sha256": descriptor_hashes[index],
        }
        for index, pixel in enumerate(pixel_rows)
    ]
    source_commit = "a" * 40
    hash_domains = METRICS.cache_gate_authority()["hash_domains"]
    observations_binding = {
        "path": "observations.npz",
        "bytes": 101,
        "sha256": "b" * 64,
    }
    determinism = {
        "schema": "occluded_goal_topological_belief_v2.encoding_determinism_receipt.v1",
        "experiment_id": E.EXPERIMENT_ID,
        "source_freeze_commit": source_commit,
        "observations_binding": observations_binding,
        "pixel_index_content_digest": "c" * 64,
        "encoder_binding": {"synthetic": True},
        "hash_domains": hash_domains,
        "pre_outcome_boundary": {
            "boundary": "IMMEDIATELY_BEFORE_FIRST_CANONICAL_ENCODER_INITIALIZATION",
            "present_v2_leaf_names": sorted((E.CONTRACT_FILE, *E.V1_REUSABLE_FILES)),
            "forbidden_outcome_leaf_names": [
                E.CALIBRATION_FILE,
                E.STAGE_A_BELIEFS_FILE,
                E.STAGE_A_METRICS_FILE,
                E.STAGE_B_TRACE_FILE,
                E.STAGE_B_METRICS_FILE,
                *E.FINAL_PUBLICATION_FILES,
            ],
            "observed_forbidden_outcome_leaf_names": [],
            "calibration_outcome_documents_opened": 0,
            "heldout_outcome_documents_opened": 0,
            "external_regeneration_receipt_present": False,
        },
        "counts": {
            "template_rows": E.EXPECTED_TEMPLATE_COUNT,
            "unique_pixel_rows": E.EXPECTED_UNIQUE_PIXEL_COUNT,
            "reused_template_rows": E.EXPECTED_REUSE_COUNT,
            "pass_count": 2,
            "encoder_invocations_per_pass": E.EXPECTED_UNIQUE_PIXEL_COUNT,
            "singleton_batch_size": 1,
        },
        "passes": [
            {
                "pass_index": pass_index,
                "fresh_encoder_instance_id": f"encoder-{pass_index}",
                "ordered_pixel_sha256s": pixel_shas,
                "records": copy.deepcopy(records),
                "canonical_cache_content_digest": E.canonical_digest(records),
            }
            for pass_index in (1, 2)
        ],
        "comparisons": {
            "pixel_order_exact": True,
            "preprocessed_tensors_exact": True,
            "raw_tokens_exact": True,
            "spatial_descriptors_exact": True,
            "canonical_cache_content_digest_exact": True,
            "pass": True,
        },
    }
    encoding_binding = {
        "path": "encoding_determinism_receipt.json",
        "bytes": len(E.canonical_document_bytes(determinism)),
        "sha256": hashlib.sha256(E.canonical_document_bytes(determinism)).hexdigest(),
    }
    v1_authority = METRICS.v1_retained_root_authority()
    by_leaf = {row["path"]: row for row in v1_authority["leaves"]}
    copies = [by_leaf[name] for name in reversed(E.V1_REUSABLE_FILES)]
    gate_names = METRICS.cache_gate_authority()["fields"]["cache_gates"]
    integrity = {
        "schema": "occluded_goal_topological_belief_v2.cache_integrity_receipt.v1",
        "experiment_id": E.EXPERIMENT_ID,
        "source_freeze_commit": source_commit,
        "v1_retained_root_binding": v1_authority,
        "copied_v1_input_bindings": copies,
        "pixel_index_content_digest": "c" * 64,
        "template_index_content_digest": "d" * 64,
        "occurrence_index_content_digest": "e" * 64,
        "encoding_determinism_receipt_binding": encoding_binding,
        "canonical_latent_index_content_digest": "f" * 64,
        "canonical_descriptor_index_content_digest": "0" * 64,
        "counts": {
            "template_rows": E.EXPECTED_TEMPLATE_COUNT,
            "unique_pixel_rows": E.EXPECTED_UNIQUE_PIXEL_COUNT,
            "reused_template_rows": E.EXPECTED_REUSE_COUNT,
            "occurrence_rows": E.EXPECTED_OCCURRENCE_COUNT,
            "canonical_token_rows": E.EXPECTED_UNIQUE_PIXEL_COUNT,
            "canonical_descriptor_rows": E.EXPECTED_UNIQUE_PIXEL_COUNT,
            "encoder_invocations_per_pass": E.EXPECTED_UNIQUE_PIXEL_COUNT,
            "multi_template_pixel_groups": E.EXPECTED_MULTI_TEMPLATE_GROUP_COUNT,
            "singleton_pixel_groups": E.EXPECTED_UNIQUE_PIXEL_COUNT - E.EXPECTED_MULTI_TEMPLATE_GROUP_COUNT,
            "templates_in_multi_template_groups": E.EXPECTED_TEMPLATE_COUNT - (E.EXPECTED_UNIQUE_PIXEL_COUNT - E.EXPECTED_MULTI_TEMPLATE_GROUP_COUNT),
        },
        "gates": {name: True for name in gate_names},
    }
    result = E._validate_gate_receipts(
        determinism,
        integrity,
        pixel_rows=pixel_rows,
        actual_bindings={
            "observations.npz": observations_binding,
            "encoding_determinism_receipt.json": encoding_binding,
        },
        source_freeze_commit=source_commit,
        hash_domains=hash_domains,
        pixel_index_content_digest="c" * 64,
        template_index_content_digest="d" * 64,
        occurrence_index_content_digest="e" * 64,
        latent_index_content_digest="f" * 64,
        descriptor_index_content_digest="0" * 64,
        v1_authority=v1_authority,
        copied_v1_bindings=[by_leaf[name] for name in E.V1_REUSABLE_FILES],
        token_row_hashes=token_hashes,
        descriptor_row_hashes=descriptor_hashes,
        required_gate_names=gate_names,
    )
    assert result["cache_integrity_gate_passed"] is True
    post_outcome = copy.deepcopy(determinism)
    post_outcome["pre_outcome_boundary"]["calibration_outcome_documents_opened"] = 1
    with pytest.raises(E.RegenerationError, match="pre-outcome boundary"):
        E._validate_gate_receipts(
            post_outcome,
            integrity,
            pixel_rows=pixel_rows,
            actual_bindings={
                "observations.npz": observations_binding,
                "encoding_determinism_receipt.json": encoding_binding,
            },
            source_freeze_commit=source_commit,
            hash_domains=hash_domains,
            pixel_index_content_digest="c" * 64,
            template_index_content_digest="d" * 64,
            occurrence_index_content_digest="e" * 64,
            latent_index_content_digest="f" * 64,
            descriptor_index_content_digest="0" * 64,
            v1_authority=v1_authority,
            copied_v1_bindings=[by_leaf[name] for name in E.V1_REUSABLE_FILES],
            token_row_hashes=token_hashes,
            descriptor_row_hashes=descriptor_hashes,
            required_gate_names=gate_names,
        )
    self_digested = {**determinism, "content_digest": "1" * 64}
    with pytest.raises(E.RegenerationError, match="must not contain a self digest"):
        E._validate_gate_receipts(
            self_digested,
            integrity,
            pixel_rows=pixel_rows,
            actual_bindings={
                "observations.npz": observations_binding,
                "encoding_determinism_receipt.json": encoding_binding,
            },
            source_freeze_commit=source_commit,
            hash_domains=hash_domains,
            pixel_index_content_digest="c" * 64,
            template_index_content_digest="d" * 64,
            occurrence_index_content_digest="e" * 64,
            latent_index_content_digest="f" * 64,
            descriptor_index_content_digest="0" * 64,
            v1_authority=v1_authority,
            copied_v1_bindings=[by_leaf[name] for name in E.V1_REUSABLE_FILES],
            token_row_hashes=token_hashes,
            descriptor_row_hashes=descriptor_hashes,
            required_gate_names=gate_names,
        )


def test_external_receipt_rejects_all_self_digest_spellings() -> None:
    for field in ("content_digest", "self_digest", "document_sha256"):
        with pytest.raises(E.RegenerationError, match="must not contain a self digest"):
            E._validate_no_self_digest({"pass": True, field: "a" * 64}, "receipt")
