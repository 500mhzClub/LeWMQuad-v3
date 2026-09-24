#!/usr/bin/env python3
"""Bind the independently reviewed V4 source closure to narrow fit licenses.

This stdlib-only metadata operation never opens dataset, RGB, target,
checkpoint, result, or metric-receipt payloads. Re-running it validates the
already-bound result and makes no further changes.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Mapping


ROOT = Path(__file__).resolve().parents[1]
METRIC_PATH = ROOT / (
    "docs/lewm_go2_observable_camera_ray_fit_v4_"
    "metric_verifier_authorization_2026-07-12.json"
)
REVIEW_PATH = ROOT / (
    "docs/lewm_go2_observable_camera_ray_fit_v4_"
    "trainer_review_record_2026-07-12.json"
)
TRAINER_PATH = ROOT / (
    "docs/lewm_go2_observable_camera_ray_fit_v4_"
    "trainer_authorization_bound_2026-07-12.json"
)
VERIFIER_PATH = ROOT / "scripts/verify_go2_observable_camera_ray_fit_v4_metrics.py"

REVIEWER = "/root/v4_final_independent_review"
REVIEWED_PENDING_AUTH_FILE_SHA256 = (
    "38b58b8f119347d520f16761cad56ead80bc2be9e4293a8b40f62c296d537d47"
)
REVIEWED_PENDING_AUTH_CONTENT_SHA256 = (
    "21cae4a1eb986e103de2a47ec24d5c650fb48b0213e7c781abfe43a1cde42ca1"
)
REVIEWED_PENDING_SOURCE_MAP_SHA256 = (
    "0cf65c798edde164de273ca7f609a9f89eba51cbf5422b01583f4afbc7efa027"
)
REVIEWED_PENDING_REVIEW_FILE_SHA256 = (
    "db23289f1b9cad5d1ea7d5d448c2068b5d6db36f26ef6b3b1445ad54ad9849e3"
)
REVIEWED_PENDING_METRIC_FILE_SHA256 = (
    "3c81be6eb58c84411572ac1e5305c54c550562f824e7044b6dd74ede677728d7"
)
REVIEWED_VERIFIER_FILE_SHA256 = (
    "c07bc01cdd70379a4829da752cc38888252070f625d83eb41d07d5b69318ec2b"
)
OLD_METRIC_CONTENT_SHA256 = (
    "673aba29a65fce8e203fcedb65f6147a1309e02c313b97cb581d52e83780fb67"
)
REVIEWED_FINAL_AUTH_FILE_SHA256 = (
    "c3fe277898f8247b630c554735e3b3ee6663dda78fd0b57f480cfe44b1ac4729"
)
REVIEWED_FINAL_AUTH_CONTENT_SHA256 = (
    "4c14f514ec784208c75b7f6c5c0779e7cbd55818cd53ed47fd09a2eb27904f80"
)
REVIEWED_FINAL_SOURCE_MAP_SHA256 = (
    "40d9c7dff078d3942e19d2047f37015e5052b866fef681fb7a624540fa1f3ed6"
)
REVIEWED_FINAL_REVIEW_FILE_SHA256 = (
    "6009d44dd0ae9ce55627728c1c157f40671eb07112144231c0ef170e31120aa0"
)
REVIEWED_FINAL_REVIEW_CONTENT_SHA256 = (
    "429cb4a936ff9186bf8463ef3970493266cf40d31ce24c36d16a529b114ca339"
)
REVIEWED_FINAL_METRIC_FILE_SHA256 = (
    "091d26f6be0372c003528be370028e6f431bcdef9770ce3855d8b1cf4045a3cf"
)
REVIEWED_FINAL_METRIC_CONTENT_SHA256 = (
    "c4090f47b417d5766f5d5100615b2f1c3891a8340e2813ad089bf894beeb98d2"
)
REVIEWED_FINAL_VERIFIER_FILE_SHA256 = (
    "d106a85ea08c2335d0816316c970b31cfaf9842874dd98c6924abaa8077d9b89"
)
PROVENANCE_REMEDIATION_SOURCE_SHA256 = {
    "scripts/train_go2_observable_camera_ray_fit_v4.py": (
        "70255e4bece10af7a1736887614e24d6cf1bdd6cc8da5c40cdf74570b2ea21d3"
    ),
    "lewm/tests/test_train_go2_observable_camera_ray_fit_v4.py": (
        "ff64aa071661a40e5f2dc1118cce60755d3a4dc7286c0a40dd0bd1fe82a42f1f"
    ),
}
LADDER_V3_INPUT_AUTH_FILE_SHA256 = (
    "11a9a4ea6274d5c02194a8ec6de4465ede00c699ec1f4fbf792cf0ebb0354255"
)
LADDER_V3_INPUT_AUTH_CONTENT_SHA256 = (
    "3655916d33f561d91a48c7c884537e31884a2782e6f58e3f3df76a9b9fd59810"
)
LADDER_V3_INPUT_SOURCE_MAP_SHA256 = (
    "084509f97ef6dc95a24877ff3205b26b88bad9595dca3f168cf76376655cd2f1"
)
LADDER_V3_INPUT_REVIEW_FILE_SHA256 = (
    "61fee8fbc4a356ca772af9dc41213ce4ad8a1426ef8059f9f9e1223f29e8c8c6"
)
LADDER_V3_INPUT_REVIEW_CONTENT_SHA256 = (
    "b533610e3f5ca9e8831392f1c6ce85a0d666e18edd2ae5e234ca5855d74f3684"
)
LADDER_V3_AMENDMENT_RELATIVE = (
    "docs/lewm_go2_observable_camera_ray_fit_v4_"
    "ladder_v3_failure_successor_amendment_2026-07-13.md"
)
LADDER_V3_AMENDMENT_SHA256 = (
    "86718d072fe151b9419318c204d4130147e098150d4fd80557f9d5865dc8f9f3"
)
LADDER_V3_SOURCE_SHA256 = {
    "lewm/benchmarks/go2_observable_camera_ray_fit_v4_ladder_gate.py": "aa51413edfea10a2d7c04b034033c83c78c27b1c08d2be1413f5917dc32e36ad",
    "lewm/tests/test_finalize_go2_observable_camera_ray_fit_v4_ladder.py": "987a50b4027b291b3c451f621063cbf64b6f8e3ab801d51314c171c7f15638d5",
    "lewm/tests/test_go2_observable_camera_ray_fit_v4_ladder_gate.py": "5a57d05d8ab13bb6366534366ea761af04386669526c159069a84a2140f478f5",
    "lewm/tests/test_launch_go2_observable_camera_ray_fit_v4.py": "3fa361313ae040c46a8c8ef5c276d12a4d116d21385c7b1462b30e1d4021c86b",
    "lewm/tests/test_train_go2_observable_camera_ray_fit_v4.py": "6274b2d6c67e3cade0baee390191f4baf1a7574abc26cb4c576adbbcf1dff91a",
    "lewm/tests/test_verify_go2_observable_camera_ray_fit_v4_metrics.py": "9d80c05f8523ab18045742ae20b183023e0a105a4bdea2fd6f9f3fd2f3d7acf1",
    "scripts/finalize_go2_observable_camera_ray_fit_v4_ladder.py": "375b1dcd3a548cf7b130fb67291ef5116effcc0197a28be42643bfc59e710ec6",
    "scripts/launch_go2_observable_camera_ray_fit_v4.py": "71d95ae79cd90c64bee8b06f2787b336d72e2fca1e23fcb0cc52f921350a2ff4",
    "scripts/train_go2_observable_camera_ray_fit_v4.py": "299980cdcb5ef561102f325bbb3db3dfd7aa8217b8a45446b0437badb8f27cfa",
    "scripts/verify_go2_observable_camera_ray_fit_v4_metrics.py": "235f7a6e2cabeaa2ff68c09c82894f69c9bfd47af0bea687dbaec5b06f27f67f",
}
V1_RESERVATION_FILE_SHA256 = (
    "115e3a4e0ad7db7f5bd6b01c7ddde29d79563600ffb84ef77a0c585f009e854e"
)
V1_FAILURE_FILE_SHA256 = (
    "6eb1becc195165e5fb49c1d222cac301f4169f301a48245d23a2b8213363af48"
)

TRAINER_FLAGS = {
    "development_fit": True,
    "development_checkpoint_creation_authorized": True,
    "checkpoint_use_authorized": False,
    "holdout_authorized": False,
    "g2_authorized": False,
    "runtime_authorized": False,
    "promotion_authorized": False,
}
METRIC_LICENSES = {
    "authorizes_verification_only_checkpoint_use": True,
    "authorizes_selected_train_target_access": True,
    "authorizes_selected_train_rgb_access": True,
    "authorizes_model_inference": True,
    "authorizes_metric_receipt_creation": True,
    "authorizes_holdout": False,
    "authorizes_g2": False,
    "authorizes_runtime": False,
    "authorizes_promotion": False,
}


def canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
        ensure_ascii=True,
    ).encode("utf-8")


def sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def load_canonical(path: Path) -> tuple[dict[str, Any], bytes]:
    raw = path.read_bytes()
    value = json.loads(raw)
    if not isinstance(value, dict) or raw != canonical_bytes(value) + b"\n":
        raise ValueError(f"metadata is not canonical: {path}")
    core = dict(value)
    declared = core.pop("content_sha256", None)
    if declared != sha256(canonical_bytes(core)):
        raise ValueError(f"metadata content hash changed: {path}")
    return value, raw


def with_content_hash(core: Mapping[str, Any]) -> tuple[dict[str, Any], bytes]:
    value = dict(core)
    value["content_sha256"] = sha256(canonical_bytes(core))
    return value, canonical_bytes(value) + b"\n"


def source_map_with(
    source_map: Mapping[str, Any],
    *,
    prospective: Mapping[Path, bytes],
    additions: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    entries = source_map.get("entries")
    if (
        source_map.get("algorithm") != "canonical_json_sha256_entries_v1"
        or not isinstance(entries, list)
        or source_map.get("entry_count") != len(entries)
    ):
        raise PermissionError("reviewed V4 source-map shape changed")
    paths = [entry.get("path") for entry in entries if isinstance(entry, dict)]
    if paths != sorted(paths) or len(set(paths)) != len(paths):
        raise PermissionError("reviewed V4 source paths are not sorted and unique")
    combined = list(entries)
    for relative, role in (additions or {}).items():
        if relative in paths:
            raise PermissionError("V4 source-map addition already exists")
        combined.append({"path": relative, "role": role, "sha256": "0" * 64})
    rebound = []
    for entry in sorted(combined, key=lambda item: str(item["path"])):
        if set(entry) != {"path", "role", "sha256"}:
            raise PermissionError("reviewed V4 source-map entry shape changed")
        path = (ROOT / entry["path"]).resolve(strict=True)
        path.relative_to(ROOT)
        payload = prospective.get(path, path.read_bytes())
        rebound.append(
            {
                "path": entry["path"],
                "role": entry["role"],
                "sha256": sha256(payload),
            }
        )
    return {
        "algorithm": "canonical_json_sha256_entries_v1",
        "entry_count": len(rebound),
        "entries": rebound,
        "source_map_sha256": sha256(canonical_bytes(rebound)),
    }


def atomic_replace(payloads: Mapping[Path, bytes]) -> None:
    temporary: dict[Path, Path] = {}
    try:
        for path, payload in payloads.items():
            candidate = path.with_name(f".{path.name}.binding-{os.getpid()}.tmp")
            with candidate.open("xb") as stream:
                stream.write(payload)
                stream.flush()
                os.fsync(stream.fileno())
            temporary[path] = candidate
        for path in payloads:
            os.replace(temporary[path], path)
        directory_fd = os.open(ROOT / "docs", os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        for candidate in temporary.values():
            candidate.unlink(missing_ok=True)


def validate_final() -> dict[str, object]:
    metric, metric_raw = load_canonical(METRIC_PATH)
    review, review_raw = load_canonical(REVIEW_PATH)
    trainer, trainer_raw = load_canonical(TRAINER_PATH)
    verifier_raw = VERIFIER_PATH.read_bytes()
    if (
        metric.get("status") != "authorized_after_independent_review"
        or metric.get("authoritative") is not False
        or metric.get("licenses") != METRIC_LICENSES
        or metric.get("review")
        != {
            "independent_reviewer": REVIEWER,
            "review_completed": True,
            "source_closure_approved": True,
            "target_partition_constants_approved": True,
        }
        or review.get("status") != "independent_review_passed"
        or review.get("decision") != "pass"
        or review.get("reviewer") != REVIEWER
        or review.get("restricted_payload_opened") is not False
        or trainer.get("status") != "independent_review_passed_authorized"
        or trainer.get("authorization") != TRAINER_FLAGS
        or trainer.get("review_record")
        != {
            "path": str(REVIEW_PATH.resolve()),
            "file_sha256": sha256(review_raw),
            "content_sha256": review["content_sha256"],
            "status": "independent_review_passed",
        }
    ):
        raise PermissionError("final V4 license or review binding is not exact")
    actual_map = source_map_with(trainer["source_map"], prospective={})
    if trainer["source_map"] != actual_map or (
        review.get("reviewed_source_map_sha256")
        != actual_map["source_map_sha256"]
    ):
        raise PermissionError("final V4 source-map/review binding changed")
    metric_file_sha = sha256(metric_raw)
    if (
        metric_file_sha.encode("ascii") not in verifier_raw
        or str(metric["content_sha256"]).encode("ascii") not in verifier_raw
    ):
        raise PermissionError("verifier does not bind final metric authorization")
    return {
        "trainer_authorization_file_sha256": sha256(trainer_raw),
        "trainer_authorization_content_sha256": trainer["content_sha256"],
        "source_map_sha256": actual_map["source_map_sha256"],
        "review_record_file_sha256": sha256(review_raw),
        "review_record_content_sha256": review["content_sha256"],
        "metric_authorization_file_sha256": metric_file_sha,
        "metric_authorization_content_sha256": metric["content_sha256"],
        "verifier_file_sha256": sha256(verifier_raw),
        "trainer_true_fields": sorted(
            key for key, value in TRAINER_FLAGS.items() if value is True
        ),
        "metric_true_fields": sorted(
            key for key, value in METRIC_LICENSES.items() if value is True
        ),
        "all_forbidden_licenses_false": True,
    }


def bind_provenance_remediation(
    trainer: Mapping[str, Any],
    trainer_raw: bytes,
) -> dict[str, object]:
    """Rebind only the reviewed frozen-provenance typo and its regression."""

    metric, metric_raw = load_canonical(METRIC_PATH)
    review, review_raw = load_canonical(REVIEW_PATH)
    verifier_raw = VERIFIER_PATH.read_bytes()
    if (
        sha256(trainer_raw) != REVIEWED_FINAL_AUTH_FILE_SHA256
        or trainer.get("content_sha256") != REVIEWED_FINAL_AUTH_CONTENT_SHA256
        or trainer.get("source_map", {}).get("source_map_sha256")
        != REVIEWED_FINAL_SOURCE_MAP_SHA256
        or trainer.get("status") != "independent_review_passed_authorized"
        or trainer.get("authorization") != TRAINER_FLAGS
        or sha256(review_raw) != REVIEWED_FINAL_REVIEW_FILE_SHA256
        or review.get("content_sha256") != REVIEWED_FINAL_REVIEW_CONTENT_SHA256
        or sha256(metric_raw) != REVIEWED_FINAL_METRIC_FILE_SHA256
        or metric.get("content_sha256") != REVIEWED_FINAL_METRIC_CONTENT_SHA256
        or metric.get("licenses") != METRIC_LICENSES
        or sha256(verifier_raw) != REVIEWED_FINAL_VERIFIER_FILE_SHA256
    ):
        raise PermissionError("provenance remediation input is not the reviewed final binding")

    entries = trainer["source_map"]["entries"]
    changed = {}
    for entry in entries:
        path = (ROOT / entry["path"]).resolve(strict=True)
        actual = sha256(path.read_bytes())
        if actual != entry["sha256"]:
            changed[str(entry["path"])] = actual
    if changed != PROVENANCE_REMEDIATION_SOURCE_SHA256:
        raise PermissionError("provenance remediation source delta is not exact")

    final_source_map = source_map_with(trainer["source_map"], prospective={})
    review_core = dict(review)
    review_core.pop("content_sha256")
    review_core["reviewed_source_map_sha256"] = final_source_map["source_map_sha256"]
    review_core["findings"] = [
        "Independent review PASS found no blocker in the post-fixed-graph source closure and established the narrow license basis.",
        "The first N5 preflight exposed a one-byte trainer provenance typo before reservation, target/RGB payload, GPU, or training work; no attempt directory was created.",
        "The remediation changes only the frozen geometry-manifest constant and a focused pre-reservation regression; all reviewed narrow license fields are unchanged.",
        "This exact provenance-remediation candidate requires different-agent byte review before N5 execution.",
    ]
    review_bound, review_bound_raw = with_content_hash(review_core)

    trainer_core = dict(trainer)
    trainer_core.pop("content_sha256")
    trainer_core["source_map"] = final_source_map
    trainer_core["review_record"] = {
        "path": str(REVIEW_PATH.resolve()),
        "file_sha256": sha256(review_bound_raw),
        "content_sha256": review_bound["content_sha256"],
        "status": "independent_review_passed",
    }
    _trainer_bound, trainer_bound_raw = with_content_hash(trainer_core)
    atomic_replace({REVIEW_PATH: review_bound_raw, TRAINER_PATH: trainer_bound_raw})
    return validate_final()


def bind_ladder_v3_successor(
    trainer: Mapping[str, Any],
    trainer_raw: bytes,
) -> dict[str, object]:
    """Bind the exact V2-root successor without accessing training payloads."""

    metric, metric_raw = load_canonical(METRIC_PATH)
    review, review_raw = load_canonical(REVIEW_PATH)
    if (
        sha256(trainer_raw) != LADDER_V3_INPUT_AUTH_FILE_SHA256
        or trainer.get("content_sha256") != LADDER_V3_INPUT_AUTH_CONTENT_SHA256
        or trainer.get("source_map", {}).get("source_map_sha256")
        != LADDER_V3_INPUT_SOURCE_MAP_SHA256
        or trainer.get("authorization") != TRAINER_FLAGS
        or sha256(review_raw) != LADDER_V3_INPUT_REVIEW_FILE_SHA256
        or review.get("content_sha256") != LADDER_V3_INPUT_REVIEW_CONTENT_SHA256
        or sha256(metric_raw) != REVIEWED_FINAL_METRIC_FILE_SHA256
        or metric.get("content_sha256") != REVIEWED_FINAL_METRIC_CONTENT_SHA256
        or metric.get("licenses") != METRIC_LICENSES
    ):
        raise PermissionError("ladder-v3 input is not the exact reviewed V1 binding")

    v1_attempt = ROOT / (
        ".generated/go2_observable_camera_ray_fit_v4/development_fit_v1/"
        "attempts/seed_20260710/n5"
    )
    v2_root = ROOT / (
        ".generated/go2_observable_camera_ray_fit_v4/development_fit_v2"
    )
    if (
        v2_root.exists()
        or not v1_attempt.is_dir()
        or sorted(path.name for path in v1_attempt.iterdir())
        != ["failed.json", "reservation.json"]
        or sha256((v1_attempt / "reservation.json").read_bytes())
        != V1_RESERVATION_FILE_SHA256
        or sha256((v1_attempt / "failed.json").read_bytes())
        != V1_FAILURE_FILE_SHA256
    ):
        raise PermissionError("immutable V1 failure or clean V2 root changed")
    amendment = ROOT / LADDER_V3_AMENDMENT_RELATIVE
    if sha256(amendment.read_bytes()) != LADDER_V3_AMENDMENT_SHA256:
        raise PermissionError("ladder-v3 amendment changed")

    changed = {}
    for entry in trainer["source_map"]["entries"]:
        path = (ROOT / entry["path"]).resolve(strict=True)
        actual = sha256(path.read_bytes())
        if actual != entry["sha256"]:
            changed[str(entry["path"])] = actual
    if changed != LADDER_V3_SOURCE_SHA256:
        raise PermissionError("ladder-v3 source delta is not exact")

    final_source_map = source_map_with(
        trainer["source_map"],
        prospective={},
        additions={
            LADDER_V3_AMENDMENT_RELATIVE: (
                "ladder_v3_failure_successor_amendment"
            )
        },
    )
    review_core = dict(review)
    review_core.pop("content_sha256")
    review_core["reviewed_source_map_sha256"] = final_source_map["source_map_sha256"]
    review_core["findings"] = [
        "The prior narrow development-fit and metric licenses are unchanged.",
        "V1 N5 is an immutable terminal failure bound by exact reservation and failure hashes; V2 uses a separate clean development_fit_v2 root.",
        "The only warning-policy change normalizes one optional exact PyTorch Context.cpp positive-decimal source trailer after a byte-exact allowlisted warning and retains raw plus normalized evidence.",
        "Rungs, seeds, steps, data, target partitions, model, thresholds, and license fields are unchanged; no result-derived tuning was performed.",
        "This exact ladder-v3 successor requires different-agent byte review before V2 N5 execution.",
    ]
    review_bound, review_bound_raw = with_content_hash(review_core)

    trainer_core = dict(trainer)
    trainer_core.pop("content_sha256")
    fit_contract = dict(trainer_core["fit_contract"])
    fit_contract.update(
        {
            "ladder_contract": "observable_camera_ray_fit_v4_ladder_v3",
            "development_output_root": (
                ".generated/go2_observable_camera_ray_fit_v4/development_fit_v2"
            ),
            "ladder_v3_amendment_file_sha256": LADDER_V3_AMENDMENT_SHA256,
            "v1_failure_lineage": {
                "reservation_file_sha256": V1_RESERVATION_FILE_SHA256,
                "reservation_content_sha256": (
                    "ca458f9371a211017f1b7a710b41508e2219a1afe19516ace2553a8eaa4d15dd"
                ),
                "failure_file_sha256": V1_FAILURE_FILE_SHA256,
                "failure_content_sha256": (
                    "7c1fe8f1ea73d8caef33debd9076bc3ddcacfaf337ec2a0000cec64f678c21e4"
                ),
            },
        }
    )
    trainer_core["fit_contract"] = fit_contract
    trainer_core["source_map"] = final_source_map
    trainer_core["review_record"] = {
        "path": str(REVIEW_PATH.resolve()),
        "file_sha256": sha256(review_bound_raw),
        "content_sha256": review_bound["content_sha256"],
        "status": "independent_review_passed",
    }
    _trainer_bound, trainer_bound_raw = with_content_hash(trainer_core)
    atomic_replace({REVIEW_PATH: review_bound_raw, TRAINER_PATH: trainer_bound_raw})
    return validate_final()


def main() -> int:
    trainer, trainer_raw = load_canonical(TRAINER_PATH)
    if trainer.get("status") == "independent_review_passed_authorized":
        try:
            result = validate_final()
        except PermissionError:
            try:
                result = bind_ladder_v3_successor(trainer, trainer_raw)
            except PermissionError:
                result = bind_provenance_remediation(trainer, trainer_raw)
        print(json.dumps(result, sort_keys=True))
        return 0
    metric, metric_raw = load_canonical(METRIC_PATH)
    review, review_raw = load_canonical(REVIEW_PATH)
    verifier_raw = VERIFIER_PATH.read_bytes()
    if (
        sha256(trainer_raw) != REVIEWED_PENDING_AUTH_FILE_SHA256
        or trainer.get("content_sha256") != REVIEWED_PENDING_AUTH_CONTENT_SHA256
        or trainer.get("source_map", {}).get("source_map_sha256")
        != REVIEWED_PENDING_SOURCE_MAP_SHA256
        or trainer.get("status") != "pending_independent_review"
        or any(value is not False for value in trainer["authorization"].values())
        or sha256(review_raw) != REVIEWED_PENDING_REVIEW_FILE_SHA256
        or review.get("status") != "pending_second_independent_review"
        or sha256(metric_raw) != REVIEWED_PENDING_METRIC_FILE_SHA256
        or metric.get("status") != "pending_independent_review"
        or any(value is not False for value in metric["licenses"].values())
        or sha256(verifier_raw) != REVIEWED_VERIFIER_FILE_SHA256
    ):
        raise PermissionError("inputs differ from independently reviewed all-false closure")

    metric_core = dict(metric)
    metric_core.pop("content_sha256")
    metric_core["status"] = "authorized_after_independent_review"
    metric_core["review"] = {
        "independent_reviewer": REVIEWER,
        "review_completed": True,
        "source_closure_approved": True,
        "target_partition_constants_approved": True,
    }
    metric_core["licenses"] = dict(METRIC_LICENSES)
    metric_bound, metric_bound_raw = with_content_hash(metric_core)
    metric_file_sha = sha256(metric_bound_raw)

    verifier_text = verifier_raw.decode("ascii")
    for old, new in (
        (REVIEWED_PENDING_METRIC_FILE_SHA256, metric_file_sha),
        (OLD_METRIC_CONTENT_SHA256, str(metric_bound["content_sha256"])),
    ):
        if verifier_text.count(old) != 1:
            raise PermissionError("verifier metric-authorization binding is not singular")
        verifier_text = verifier_text.replace(old, new)
    verifier_bound_raw = verifier_text.encode("ascii")

    prospective = {
        METRIC_PATH.resolve(): metric_bound_raw,
        VERIFIER_PATH.resolve(): verifier_bound_raw,
    }
    final_source_map = source_map_with(trainer["source_map"], prospective=prospective)
    review_core = {
        "schema": review["schema"],
        "status": "independent_review_passed",
        "decision": "pass",
        "reviewer": REVIEWER,
        "reviewed_source_map_sha256": final_source_map["source_map_sha256"],
        "restricted_payload_opened": False,
        "findings": [
            "Independent review PASS found no remaining blocker in the post-fixed-graph candidate.",
            "The reviewer reconstructed all 42 sorted, unique, exact-role source entries without opening restricted payloads.",
            "The final deterministic license/hash delta requires different-agent byte review before execution.",
        ],
    }
    review_bound, review_bound_raw = with_content_hash(review_core)

    trainer_core = dict(trainer)
    trainer_core.pop("content_sha256")
    trainer_core["status"] = "independent_review_passed_authorized"
    trainer_core["authorization"] = dict(TRAINER_FLAGS)
    trainer_core["source_map"] = final_source_map
    trainer_core["review_record"] = {
        "path": str(REVIEW_PATH.resolve()),
        "file_sha256": sha256(review_bound_raw),
        "content_sha256": review_bound["content_sha256"],
        "status": "independent_review_passed",
    }
    _trainer_bound, trainer_bound_raw = with_content_hash(trainer_core)

    atomic_replace(
        {
            METRIC_PATH: metric_bound_raw,
            VERIFIER_PATH: verifier_bound_raw,
            REVIEW_PATH: review_bound_raw,
            TRAINER_PATH: trainer_bound_raw,
        }
    )
    print(json.dumps(validate_final(), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
