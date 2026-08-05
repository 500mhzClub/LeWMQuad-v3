#!/usr/bin/env python3
"""Run the one-shot task-coupled recurrent physical-dynamics successor V1."""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
import hashlib
import itertools
import json
import os
from pathlib import Path
import sys
from typing import Any

import torch
import torch.nn.functional as F


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from lewm.benchmarks import (  # noqa: E402
    go2_task_coupled_recurrent_dynamics_v1 as benchmark,
)
from lewm.datasets.go2_world_model_counterfactual_pilot_v1 import (  # noqa: E402
    read_bound_rgb_bytes_v1,
)
from scripts import run_go2_grounded_dense_dino_joint_jepa_v1 as upstream  # noqa: E402


AUTHORITY_SCHEMA = "lewm_go2_task_coupled_recurrent_dynamics_v1_execution_authority_v1"
AUTHORITY_STATUS = "AUTHORIZED_ONE_TASK_COUPLED_RECURRENT_DYNAMICS_V1"
SOURCE_REVIEW_SCHEMA = "lewm_go2_task_coupled_recurrent_dynamics_v1_source_review_v1"
SOURCE_REVIEW_STATUS = "PASS_INDEPENDENT_SOURCE_REVIEW"
RESULT_SCHEMA = "lewm_go2_task_coupled_recurrent_dynamics_v1_result_v1"
TERMINAL_SCHEMA = "lewm_go2_task_coupled_recurrent_dynamics_v1_terminal_v1"

PREREGISTRATION = REPO_ROOT / (
    "docs/lewm_go2_task_coupled_recurrent_dynamics_v1_preregistration_2026-08-04.md"
)
SOURCE_REVIEW = REPO_ROOT / (
    "docs/lewm_go2_task_coupled_recurrent_dynamics_v1_source_review_2026-08-04.json"
)
EXECUTION_AUTHORITY = REPO_ROOT / (
    "docs/lewm_go2_task_coupled_recurrent_dynamics_v1_execution_authority_2026-08-04.json"
)
DEFAULT_OUTPUT_ROOT = REPO_ROOT / (
    ".generated/dev/go2_task_coupled_recurrent_dynamics_v1/attempt_v1"
)
DINO_REPOSITORY = Path(
    "/home/andrewknowles/.cache/dinov2-7764ea0f912e53c92e82eb78a2a1631e92725fc8"
)
DINO_CHECKPOINT = Path(
    "/home/andrewknowles/.cache/torch/hub/checkpoints/dinov2_vits14_pretrain.pth"
)

SOURCE_PATHS = {
    **{
        f"grounded_upstream_{name}": path
        for name, path in upstream.SOURCE_PATHS.items()
    },
    "recurrent_model": REPO_ROOT
    / "lewm/models/go2_task_coupled_recurrent_dynamics_v1.py",
    "recurrent_benchmark": REPO_ROOT
    / "lewm/benchmarks/go2_task_coupled_recurrent_dynamics_v1.py",
    "recurrent_runner": Path(__file__).resolve(),
    "recurrent_model_test": REPO_ROOT
    / "lewm/tests/test_go2_task_coupled_recurrent_dynamics_v1.py",
    "recurrent_benchmark_test": REPO_ROOT
    / "lewm/tests/test_go2_task_coupled_recurrent_dynamics_v1_benchmark.py",
    "recurrent_runner_test": REPO_ROOT
    / "lewm/tests/test_run_go2_task_coupled_recurrent_dynamics_v1.py",
}


class RecurrentRunnerError(RuntimeError):
    """Raised when one-shot authority, access, or output contracts change."""


def canonical_bytes_v1(value: object) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")


def file_binding_v1(path: Path) -> dict[str, object]:
    selected = upstream.safe_path_v1(path, label="bound file")
    if not selected.is_file():
        raise RecurrentRunnerError("bound path is not a regular file")
    digest = hashlib.sha256()
    size = 0
    with selected.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            size += len(chunk)
            digest.update(chunk)
    return {"path": str(selected), "sha256": digest.hexdigest(), "byte_count": size}


def _require_binding(
    value: object, *, label: str, rehash: bool = True
) -> dict[str, object]:
    if (
        not isinstance(value, Mapping)
        or set(value) != {"path", "sha256", "byte_count"}
        or not isinstance(value.get("path"), str)
        or not isinstance(value.get("sha256"), str)
        or len(str(value["sha256"])) != 64
        or type(value.get("byte_count")) is not int
        or int(value["byte_count"]) <= 0
    ):
        raise RecurrentRunnerError(f"{label} binding is malformed")
    observed = dict(value)
    upstream._reject_protected(  # noqa: SLF001
        Path(str(observed["path"])), label=label
    )
    if rehash and file_binding_v1(Path(str(observed["path"]))) != observed:
        raise RecurrentRunnerError(f"{label} binding changed")
    return observed


def _read_json_binding(value: object, *, label: str) -> dict[str, Any]:
    binding = _require_binding(value, label=label)
    raw = Path(str(binding["path"])).read_bytes()
    try:
        document = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise RecurrentRunnerError(f"{label} is not strict JSON") from error
    if not isinstance(document, dict):
        raise RecurrentRunnerError(f"{label} must be a JSON object")
    return document


def _write_json_exclusive(path: Path, value: Mapping[str, Any]) -> None:
    raw = json.dumps(value, indent=2, sort_keys=True, allow_nan=False).encode("utf-8") + b"\n"
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    descriptor = os.open(path, flags, 0o600)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(raw)
            handle.flush()
            os.fsync(handle.fileno())
    except Exception:
        try:
            os.close(descriptor)
        except OSError:
            pass
        raise


def _save_checkpoint_exclusive(path: Path, checkpoint: Mapping[str, Any]) -> None:
    if path.exists():
        raise RecurrentRunnerError("checkpoint path already exists")
    with path.open("xb") as handle:
        torch.save(dict(checkpoint), handle)
        handle.flush()
        os.fsync(handle.fileno())
    reopened = torch.load(path, map_location="cpu", weights_only=True)
    if (
        not isinstance(reopened, Mapping)
        or reopened.get("identity_sha256")
        != benchmark.checkpoint_identity_v1(reopened)
    ):
        raise RecurrentRunnerError("checkpoint round trip changed")


def _validate_authority_v1(
    authority_path: Path, *, expected_sha256: str, expected_byte_count: int
) -> tuple[dict[str, Any], dict[str, object]]:
    expected = {
        "path": str(upstream.safe_path_v1(authority_path, label="authority")),
        "sha256": expected_sha256,
        "byte_count": expected_byte_count,
    }
    authority = _read_json_binding(expected, label="execution authority")
    if (
        authority.get("schema") != AUTHORITY_SCHEMA
        or authority.get("status") != AUTHORITY_STATUS
        or authority.get("output_root") != str(DEFAULT_OUTPUT_ROOT.resolve())
        or authority.get("config") != benchmark.config_v1()
    ):
        raise RecurrentRunnerError("execution authority contract changed")
    prereg = _require_binding(
        authority.get("preregistration_binding"), label="preregistration"
    )
    if prereg["path"] != str(PREREGISTRATION.resolve()):
        raise RecurrentRunnerError("preregistration path changed")
    review_binding = _require_binding(
        authority.get("source_review_binding"), label="source review"
    )
    if review_binding["path"] != str(SOURCE_REVIEW.resolve()):
        raise RecurrentRunnerError("source review path changed")
    sources = authority.get("source_bindings")
    if not isinstance(sources, Mapping) or set(sources) != set(SOURCE_PATHS):
        raise RecurrentRunnerError("source closure changed")
    for name, path in SOURCE_PATHS.items():
        binding = _require_binding(sources[name], label=f"source {name}")
        if binding["path"] != str(path.resolve()):
            raise RecurrentRunnerError(f"source {name} path changed")
    review = _read_json_binding(review_binding, label="source review")
    if (
        review.get("schema") != SOURCE_REVIEW_SCHEMA
        or review.get("status") != SOURCE_REVIEW_STATUS
        or review.get("protected_material_opened") is not False
        or review.get("preregistration_binding") != prereg
        or review.get("source_bindings") != sources
        or review.get("findings") != []
    ):
        raise RecurrentRunnerError("independent source review changed")
    fixed_inputs = upstream.fixed_input_bindings_v1()
    if authority.get("input_bindings") != fixed_inputs:
        raise RecurrentRunnerError("frozen input bindings changed")
    for name, binding in fixed_inputs.items():
        # Reading an evaluation role binding here would itself violate the
        # post-checkpoint permission.  Its exact declaration is frozen now and
        # the upstream role loader rehashes it after ``ledger.checkpoint()``.
        late = name in {"posthoc_eval_rows", "eval_role_index"} or name.startswith(
            "eval_state_receipt_"
        )
        _require_binding(binding, label=f"input {name}", rehash=not late)
    dino = authority.get("dino")
    expected_dino = {
        "repository_path": str(DINO_REPOSITORY.resolve()),
        "repository_commit": upstream.DINO_REPOSITORY_COMMIT,
        "checkpoint_binding": {
            "path": str(DINO_CHECKPOINT.resolve()),
            "sha256": upstream.DINO_CHECKPOINT_SHA256,
            "byte_count": upstream.DINO_CHECKPOINT_BYTE_COUNT,
        },
    }
    if dino != expected_dino:
        raise RecurrentRunnerError("frozen DINO binding changed")
    _require_binding(dino["checkpoint_binding"], label="DINO checkpoint")
    permissions = authority.get("permissions")
    if permissions != {
        "train_receipt_access": True,
        "train_context_rgb_access": True,
        "eval_receipt_access_after_checkpoint": True,
        "eval_context_rgb_access_after_checkpoint": True,
        "successor_rgb_access": False,
        "data_generation": False,
        "sealed_or_protected_access": False,
        "retry_resume_overwrite": False,
    }:
        raise RecurrentRunnerError("authority permissions changed")
    return authority, expected


@dataclass
class ContextOnlyLedgerV1:
    """Semantic access ledger with no successor-observation route."""

    stage: str = "created"
    checkpoint_durable: bool = False
    receipt_loads: dict[str, int] = field(
        default_factory=lambda: {"train": 0, "eval": 0}
    )
    role_index_opens: dict[str, int] = field(
        default_factory=lambda: {"train": 0, "eval": 0}
    )
    state_receipt_opens: dict[str, int] = field(
        default_factory=lambda: {"train": 0, "eval": 0}
    )
    rgb_opens: dict[str, int] = field(
        default_factory=lambda: {
            "train_context": 0,
            "train_successor": 0,
            "eval_context": 0,
            "eval_successor": 0,
        }
    )
    opened_receipts: set[tuple[str, str]] = field(default_factory=set)
    opened_artifacts: set[tuple[str, str]] = field(default_factory=set)

    def load_receipts(self, role: str) -> None:
        if role == "train":
            if self.stage != "created":
                raise RecurrentRunnerError("train receipts must open first")
            self.receipt_loads[role] = 1
            self.stage = "train"
        elif role == "eval":
            if not self.checkpoint_durable or self.receipt_loads[role]:
                raise RecurrentRunnerError("evaluation opened before checkpoint")
            self.receipt_loads[role] = 1
            self.stage = "eval"
        else:
            raise RecurrentRunnerError("unknown receipt role")

    def open_role_index(self, role: str, path: str) -> None:
        if self.receipt_loads.get(role) != 1 or self.role_index_opens[role]:
            raise RecurrentRunnerError("role index opened outside its stage")
        if not path:
            raise RecurrentRunnerError("role index path is empty")
        self.role_index_opens[role] = 1

    def open_state_receipt(self, role: str, path: str) -> None:
        key = (role, path)
        if self.receipt_loads.get(role) != 1 or not path or key in self.opened_receipts:
            raise RecurrentRunnerError("state receipt opened outside its stage")
        self.opened_receipts.add(key)
        self.state_receipt_opens[role] += 1

    def open_rgb(self, role: str, kind: str, artifact_id: str) -> None:
        if kind != "context":
            raise RecurrentRunnerError("successor RGB access is structurally forbidden")
        if role == "train" and self.stage != "train":
            raise RecurrentRunnerError("train context opened outside train stage")
        if role == "eval" and self.stage != "eval":
            raise RecurrentRunnerError("eval context opened outside eval stage")
        key = (role, artifact_id)
        if not artifact_id or key in self.opened_artifacts:
            raise RecurrentRunnerError("context artifact opened more than once")
        self.opened_artifacts.add(key)
        self.rgb_opens[f"{role}_context"] += 1

    def checkpoint(self) -> None:
        if self.stage != "train" or self.checkpoint_durable:
            raise RecurrentRunnerError("checkpoint durability order changed")
        self.checkpoint_durable = True
        self.stage = "checkpoint"

    def finalized(self) -> dict[str, Any]:
        audit = {
            "stage": self.stage,
            "checkpoint_durable": self.checkpoint_durable,
            "receipt_loads": dict(self.receipt_loads),
            "role_index_opens": dict(self.role_index_opens),
            "state_receipt_opens": dict(self.state_receipt_opens),
            "rgb_opens": dict(self.rgb_opens),
            "unique_context_artifacts": len(self.opened_artifacts),
            "successor_rgb_open_count": self.rgb_opens["train_successor"]
            + self.rgb_opens["eval_successor"],
        }
        if (
            self.stage != "eval"
            or not self.checkpoint_durable
            or self.receipt_loads != {"train": 1, "eval": 1}
            or self.role_index_opens != {"train": 1, "eval": 1}
            or self.state_receipt_opens != {"train": 128, "eval": 128}
            or self.rgb_opens
            != {
                "train_context": 384,
                "train_successor": 0,
                "eval_context": 384,
                "eval_successor": 0,
            }
            or audit["unique_context_artifacts"] != 768
        ):
            raise RecurrentRunnerError("context-only access accounting changed")
        return audit


@torch.inference_mode()
def _full_dino_context_tokens_v1(
    role: Any,
    *,
    ledger: ContextOnlyLedgerV1,
    dino: upstream.FrozenDINOTrunkV1,
) -> torch.Tensor:
    artifact_ids = tuple(itertools.chain.from_iterable(role.context_artifact_ids))
    trunks = upstream.precompute_trunks_v1(
        artifact_ids,
        role=role.role,
        kind="context",
        ledger=ledger,
        bound_reader=lambda artifact_id: read_bound_rgb_bytes_v1(role.bundle, artifact_id),
        trunk=dino,
    )
    rows: list[torch.Tensor] = []
    for start in range(0, len(artifact_ids), 16):
        hidden = trunks[start : start + 16].to(dino.device)
        for block in tuple(dino.dino.blocks)[10:12]:
            hidden = block(hidden)
        hidden = dino.dino.norm(hidden)
        patches = F.normalize(hidden[:, 1:], dim=-1).cpu()
        rows.append(patches)
    result = torch.cat(rows).reshape(128, 3, 256, 384).contiguous()
    return benchmark.validate_context_tokens_v1(result, role=role.role)


def _result_identity_v1(value: Mapping[str, Any]) -> str:
    document = dict(value)
    document.pop("result_identity_sha256", None)
    return hashlib.sha256(canonical_bytes_v1(document)).hexdigest()


def execute_v1(
    authority: Mapping[str, Any], *, authority_binding: Mapping[str, object]
) -> dict[str, Any]:
    output_root = upstream.safe_path_v1(
        Path(str(authority["output_root"])), label="output root", must_exist=False
    )
    output_root.mkdir(parents=True, exist_ok=False)
    _write_json_exclusive(
        output_root / "reservation.json",
        {
            "schema": "lewm_go2_task_coupled_recurrent_dynamics_v1_reservation_v1",
            "status": "CONSUMED_ONE_SHOT_ATTEMPT",
            "authority_binding": dict(authority_binding),
        },
    )
    determinism = upstream.configure_determinism_v1()
    ledger = ContextOnlyLedgerV1()
    shared = upstream._load_shared_role_metadata_v1(authority)  # noqa: SLF001
    ledger.load_receipts("train")
    train = upstream.load_role_runtime_data_v1(
        authority, shared, role="train", ledger=ledger
    )
    if not torch.cuda.is_available():
        raise RecurrentRunnerError("the authorized ROCm device is unavailable")
    device = torch.device("cuda:0")
    dino = upstream.load_dino_trunk_v1(
        Path(str(authority["dino"]["repository_path"])),
        Path(str(authority["dino"]["checkpoint_binding"]["path"])),
        device=device,
    )
    train_context = _full_dino_context_tokens_v1(train, ledger=ledger, dino=dino)
    checkpoint = benchmark.fit_checkpoint_v1(train, train_context, device=device)
    checkpoint_path = output_root / "checkpoint.pt"
    _save_checkpoint_exclusive(checkpoint_path, checkpoint)
    checkpoint_binding = file_binding_v1(checkpoint_path)
    ledger.checkpoint()
    del train_context
    torch.cuda.empty_cache()

    ledger.load_receipts("eval")
    evaluation = upstream.load_role_runtime_data_v1(
        authority, shared, role="eval", ledger=ledger
    )
    disjointness = upstream.assert_role_disjointness_v1(train.plan, evaluation.plan)
    eval_context = _full_dino_context_tokens_v1(evaluation, ledger=ledger, dino=dino)
    first = benchmark.evaluate_checkpoint_v1(
        checkpoint,
        train,
        evaluation,
        eval_context,
        device=device,
        integrity_passed=True,
    )
    second = benchmark.evaluate_checkpoint_v1(
        checkpoint,
        train,
        evaluation,
        eval_context,
        device=device,
        integrity_passed=True,
    )
    if canonical_bytes_v1(first) != canonical_bytes_v1(second):
        raise RecurrentRunnerError("repeat evaluation was not bitwise identical")
    access_audit = ledger.finalized()
    result: dict[str, Any] = {
        "schema": RESULT_SCHEMA,
        "status": first["status"],
        "citable_as_scientific_evidence": False,
        "development_only": True,
        "authority_binding": dict(authority_binding),
        "preregistration_binding": dict(authority["preregistration_binding"]),
        "checkpoint_binding": checkpoint_binding,
        "checkpoint_summary": {
            "identity_sha256": checkpoint["identity_sha256"],
            "arms": {
                arm: [
                    {
                        "seed": member["seed"],
                        "initial_state_identity_sha256": member[
                            "initial_state_identity_sha256"
                        ],
                        "state_identity_sha256": member["state_identity_sha256"],
                        "updates": member["updates"],
                        "trace": member["trace"],
                        "training_seconds": member["training_seconds"],
                    }
                    for member in checkpoint["arms"][arm]
                ]
                for arm in benchmark.ARM_ORDER
            },
        },
        "evaluation": first,
        "repeat_evaluation_exact": True,
        "role_disjointness": disjointness,
        "access_audit": access_audit,
        "runtime": {
            "determinism": determinism,
            "torch": torch.__version__,
            "hip": torch.version.hip,
            "device": torch.cuda.get_device_name(device),
            "frozen_dino": {
                "blocks": list(range(12)),
                "final_norm": True,
                "l2_normalized_patch_tokens": True,
                "trainable": False,
            },
        },
        "successor_observations_opened": 0,
        "authorizes_navigation_claim": False,
        "authorizes_blind_rollout_preregistration": first[
            "authorizes_blind_rollout_preregistration"
        ],
    }
    result["result_identity_sha256"] = _result_identity_v1(result)
    _write_json_exclusive(output_root / "result.json", result)
    terminal = {
        "schema": TERMINAL_SCHEMA,
        "status": result["status"],
        "authorizes_retry_or_resume": False,
        "authorizes_navigation_claim": False,
        "authorizes_blind_rollout_preregistration": result[
            "authorizes_blind_rollout_preregistration"
        ],
        "result_binding": file_binding_v1(output_root / "result.json"),
        "failure": None,
    }
    _write_json_exclusive(output_root / "terminal.json", terminal)
    observed = {entry.name for entry in output_root.iterdir()}
    if observed != {"reservation.json", "checkpoint.pt", "result.json", "terminal.json"}:
        raise RecurrentRunnerError("attempt output inventory changed")
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--authority", type=Path, required=True)
    parser.add_argument("--expected-authority-sha256", required=True)
    parser.add_argument("--expected-authority-byte-count", type=int, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    authority: Mapping[str, Any] | None = None
    output_root: Path | None = None
    try:
        authority, binding = _validate_authority_v1(
            args.authority,
            expected_sha256=args.expected_authority_sha256,
            expected_byte_count=args.expected_authority_byte_count,
        )
        output_root = Path(str(authority["output_root"]))
        result = execute_v1(authority, authority_binding=binding)
        print(json.dumps({"status": result["status"], "output_root": str(output_root)}))
        return 0
    except Exception as error:
        if output_root is not None and output_root.is_dir():
            terminal_path = output_root / "terminal.json"
            if not terminal_path.exists():
                try:
                    _write_json_exclusive(
                        terminal_path,
                        {
                            "schema": TERMINAL_SCHEMA,
                            "status": "FAIL_INFRASTRUCTURE_NO_SCIENTIFIC_DECISION",
                            "authorizes_retry_or_resume": False,
                            "authorizes_navigation_claim": False,
                            "authorizes_blind_rollout_preregistration": False,
                            "result_binding": None,
                            "failure": {"type": type(error).__name__, "message": str(error)},
                        },
                    )
                except Exception:
                    pass
        print(f"error: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
