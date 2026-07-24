from __future__ import annotations

import hashlib
import importlib.util
from pathlib import Path
import unittest
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[2]
CONTRACT_PATH = (
    ROOT / "lewm/benchmarks/go2_rgb_causal_motion_alignment_v1.py"
)
_SPEC = importlib.util.spec_from_file_location(
    "_test_go2_rgb_causal_motion_alignment_v1_receipt_boundary",
    CONTRACT_PATH,
)
assert _SPEC is not None and _SPEC.loader is not None
contract = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(contract)


def _reservation_binding() -> dict[str, Any]:
    return {
        "path": "reservation.json",
        "file_sha256": "1" * 64,
        "content_sha256": "2" * 64,
        "byte_count": 1,
    }


def _expected_binding(path: str) -> dict[str, Any]:
    return {
        "path": path,
        "file_sha256": hashlib.sha256(path.encode("ascii")).hexdigest(),
        "content_sha256": None,
        "byte_count": len(path),
    }


def _record(
    sequence: int,
    previous: str | None,
    record_type: str,
    **fields: Any,
) -> dict[str, Any]:
    return contract.with_content_sha256({
        "schema": contract.PARTIAL_ACCESS_RECORD_SCHEMA,
        "sequence": sequence,
        "previous_record_content_sha256": previous,
        "record_type": record_type,
        **fields,
    })


def _open_rows(
    path: str,
    *,
    purpose: str,
    role: str,
    kind: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    expected = _expected_binding(path)
    attempted = {
        "open_id": 0,
        "stage": (
            "terminal_input_rehash"
            if purpose == "terminal_rehash"
            else "synthetic_runtime_load"
        ),
        "kind": kind,
        "role": role,
        "purpose": purpose,
        "expected_binding": expected,
    }
    outcome = {
        "open_id": 0,
        "stage": attempted["stage"],
        "kind": kind,
        "outcome": "ACCEPTED",
        "descriptor_opened": True,
        "read_completed": True,
        "binding_accepted": True,
        "observed_binding": {
            "path": path,
            "file_sha256": expected["file_sha256"],
            "byte_count": expected["byte_count"],
        },
        "partial_byte_count": expected["byte_count"],
        "error": None,
    }
    return attempted, outcome


def _ledger(
    opens: Iterable[tuple[str, str, str, str]],
    *,
    finalized: bool = True,
) -> bytes:
    records: list[dict[str, Any]] = []

    def append(record_type: str, **fields: Any) -> None:
        previous = (
            records[-1]["content_sha256"] if records else None
        )
        records.append(_record(
            len(records), previous, record_type, **fields
        ))

    append(
        "LEDGER_OPENED",
        attempt_identity="a" * 64,
        reservation=_reservation_binding(),
    )
    for open_id, (path, purpose, role, kind) in enumerate(opens, start=1):
        attempted, outcome = _open_rows(
            path, purpose=purpose, role=role, kind=kind
        )
        attempted["open_id"] = open_id
        outcome["open_id"] = open_id
        append("OPEN_ATTEMPTED", **attempted)
        append("OPEN_OUTCOME", **outcome)
    terminal_fields: dict[str, Any] = {
        "stage": {
            "name": "synthetic_terminal",
            "update": None,
            "microbatch": None,
            "checkpoint_update": None,
            "role": "authority",
        },
        "operation_counts": contract.empty_partial_operation_counts(),
        "error": None,
    }
    terminal_type = "RUNTIME_INPUT_ACCESS_FINALIZED"
    if not finalized:
        terminal_type = "ATTEMPT_TERMINATING"
        message = "synthetic terminal failure"
        terminal_fields["error"] = {
            "type": "RuntimeError",
            "message": message,
            "message_sha256":
                hashlib.sha256(message.encode("utf-8")).hexdigest(),
        }
    append(terminal_type, **terminal_fields)
    return b"".join(
        contract.canonical_json_bytes(record) + b"\n"
        for record in records
    )


def _render_path() -> str:
    return (
        ".generated/go2_render_selected_v04/scenes/"
        "scene_0123456789abcdef/rgb/frame_000123_env_04.png"
    )


class MotionAlignmentReceiptBoundaryTests(unittest.TestCase):
    def test_render_rgb_runtime_load_and_terminal_rehash_fully_parse(
        self,
    ) -> None:
        path = _render_path()
        raw = _ledger([
            (path, "runtime_load", "train", "development_rgb"),
            (
                path,
                "terminal_rehash",
                "authority",
                "development_rgb",
            ),
        ])
        records = contract.validate_finalized_access_ledger(raw)
        attempted = [
            record for record in records
            if record["record_type"] == "OPEN_ATTEMPTED"
        ]
        self.assertEqual(
            [record["purpose"] for record in attempted],
            ["runtime_load", "terminal_rehash"],
        )
        self.assertTrue(all(
            record["expected_binding"]["path"] == path
            for record in attempted
        ))
        self.assertEqual(
            records[-1]["record_type"], "RUNTIME_INPUT_ACCESS_FINALIZED"
        )

    def test_exact_runner_context_table_is_admitted(self) -> None:
        fixed = [
            (contract.SCHEDULE_RELATIVE_PATH, "bound_schedule"),
            (contract.N320_GATE_RELATIVE_PATH, "n320_gate"),
            (contract.N320_CHECKPOINT_RELATIVE_PATH, "n320_checkpoint"),
            (
                contract.RAW_MANIFEST_RELATIVE_PATH,
                "raw_authority_manifest",
            ),
            (contract.RAW_AUDIT_RELATIVE_PATH, "raw_authority_audit"),
            (contract.RAW_PAIRS_RELATIVE_PATH, "raw_pairs_index"),
            (contract.RAW_ENDPOINTS_RELATIVE_PATH, "raw_endpoints_index"),
        ]
        opens = [
            (
                path,
                purpose,
                "authority",
                kind,
            )
            for path, kind in fixed
            for purpose in ("runtime_load", "terminal_rehash")
        ]
        raw_payload = (
            f"{contract.RAW_ROOT_RELATIVE_PATH}/shards/"
            "0123456789abcdef/camera_basis_body_fru.f4"
        )
        for path, kind in (
            (raw_payload, "raw_supervision"),
            (_render_path(), "development_rgb"),
        ):
            opens.extend([
                (path, "runtime_load", "train", kind),
                (
                    path,
                    "runtime_load",
                    "checkpoint_selection",
                    kind,
                ),
                (path, "terminal_rehash", "authority", kind),
            ])
        records = contract.validate_finalized_access_ledger(_ledger(opens))
        self.assertEqual(
            sum(
                record["record_type"] == "OPEN_ATTEMPTED"
                for record in records
            ),
            len(opens),
        )

    def test_near_miss_render_paths_are_rejected(self) -> None:
        near_misses = [
            _render_path().replace("0123456789abcdef", "0123456789abcdeF"),
            _render_path().replace("0123456789abcdef", "0123456789abcde"),
            _render_path().replace("frame_000123", "frame_00123"),
            _render_path().replace(".png", ".jpg"),
            _render_path() + ".bak",
            _render_path().replace(
                "go2_render_selected_v04", "go2_render_selected_v05"
            ),
            _render_path().replace("/rgb/", "/rgb_extra/"),
        ]
        for path in near_misses:
            with self.subTest(path=path):
                self.assertFalse(contract.is_development_rgb_path(path))
                with self.assertRaises(PermissionError):
                    contract.parse_partial_access_ledger(_ledger([
                        (
                            path,
                            "runtime_load",
                            "train",
                            "development_rgb",
                        ),
                    ]))

    def test_path_purpose_role_kind_crossings_are_rejected(self) -> None:
        raw_payload = (
            f"{contract.RAW_ROOT_RELATIVE_PATH}/shards/"
            "0123456789abcdef/camera_basis_body_fru.f4"
        )
        crossings = [
            (
                contract.SCHEDULE_RELATIVE_PATH,
                "runtime_load",
                "authority",
                "n320_gate",
            ),
            (
                contract.SCHEDULE_RELATIVE_PATH,
                "runtime_load",
                "train",
                "bound_schedule",
            ),
            (
                contract.N320_CHECKPOINT_RELATIVE_PATH,
                "terminal_rehash",
                "checkpoint_selection",
                "n320_checkpoint",
            ),
            (
                contract.RAW_MANIFEST_RELATIVE_PATH,
                "runtime_load",
                "authority",
                "raw_supervision",
            ),
            (
                contract.RAW_PAIRS_RELATIVE_PATH,
                "runtime_load",
                "train",
                "raw_pairs_index",
            ),
            (
                raw_payload,
                "runtime_load",
                "authority",
                "raw_supervision",
            ),
            (
                raw_payload,
                "terminal_rehash",
                "train",
                "raw_supervision",
            ),
            (
                raw_payload,
                "runtime_load",
                "train",
                "development_rgb",
            ),
            (
                _render_path(),
                "runtime_load",
                "authority",
                "development_rgb",
            ),
            (
                _render_path(),
                "terminal_rehash",
                "checkpoint_selection",
                "development_rgb",
            ),
            (
                _render_path(),
                "runtime_load",
                "train",
                "raw_supervision",
            ),
            (
                _render_path(),
                "runtime_load",
                "index",
                "development_rgb",
            ),
        ]
        for path, purpose, role, kind in crossings:
            with self.subTest(
                path=path,
                purpose=purpose,
                role=role,
                kind=kind,
            ):
                with self.assertRaisesRegex(
                    PermissionError,
                    "path/purpose/role/kind context changed",
                ):
                    contract.parse_partial_access_ledger(_ledger([
                        (path, purpose, role, kind),
                    ]))

    def test_prior_outputs_sealed_and_symlink_style_paths_reject(self) -> None:
        forbidden = [
            (
                ".generated/go2_shared_observable_camera_ray_jepa_v5/"
                "rgb_causal_temporal_perception_probe_v1/result.json"
            ),
            f"{contract.RAW_ROOT_RELATIVE_PATH}/sealed/private.bin",
            f"{contract.RAW_ROOT_RELATIVE_PATH}/sealed_legacy/private.bin",
            f"{contract.RAW_ROOT_RELATIVE_PATH}/../escape.bin",
            f"{contract.RAW_ROOT_RELATIVE_PATH}//shards/alias.bin",
            "/absolute/render.png",
        ]
        for path in forbidden:
            with self.subTest(path=path):
                with self.assertRaises(PermissionError):
                    contract.parse_partial_access_ledger(_ledger([
                        (
                            path,
                            "runtime_load",
                            "train",
                            "raw_supervision",
                        ),
                    ]))

    def test_unpaired_incomplete_and_nonfinal_ledgers_are_inadmissible(
        self,
    ) -> None:
        path = _render_path()
        valid = contract.parse_partial_access_ledger(_ledger([
            (path, "runtime_load", "train", "development_rgb"),
        ]))

        header_only = (
            contract.canonical_json_bytes(valid[0]) + b"\n"
        )
        with self.assertRaisesRegex(PermissionError, "ledger is incomplete"):
            contract.parse_partial_access_ledger(header_only)

        attempted_only = b"".join(
            contract.canonical_json_bytes(record) + b"\n"
            for record in valid[:2]
        )
        with self.assertRaisesRegex(PermissionError, "ledger is incomplete"):
            contract.parse_partial_access_ledger(attempted_only)

        terminating = _ledger([
            (path, "runtime_load", "train", "development_rgb"),
        ], finalized=False)
        parsed = contract.parse_partial_access_ledger(terminating)
        self.assertEqual(parsed[-1]["record_type"], "ATTEMPT_TERMINATING")
        with self.assertRaisesRegex(
            PermissionError, "requires a finalized access ledger"
        ):
            contract.validate_finalized_access_ledger(terminating)

        with self.assertRaises(PermissionError):
            contract.validate_finalized_access_ledger(
                _ledger([
                    (
                        path,
                        "runtime_load",
                        "train",
                        "development_rgb",
                    ),
                ]) + b"\n"
            )


if __name__ == "__main__":
    unittest.main()
