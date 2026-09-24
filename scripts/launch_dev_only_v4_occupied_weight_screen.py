#!/usr/bin/env python3
"""DEVELOPMENT-ONLY screening run: v4 O-field occupied weighting.

**NOT CLAIM BEARING.**  This launcher deliberately does not construct or consume
a v4 source review or execution authorization, and the resulting checkpoint has
not passed the reviewed execution path.  It exists to screen whether the
committed v4 5:1 occupied-to-free O-field weighting has any value before the
formal artifact chain is built.

It reuses the frozen leaf training implementation unchanged and substitutes only
``model_api`` -- the same injection point the semantic-anchor successors use.
The frozen v3 execution authorization is read **solely as a configuration
source** for corpus, N320 initialization and schedule identity; it is not
treated as a grant for this run.

Three fidelity checks run before any training step, and the run aborts unless
all three pass:

1. initial-state tensor equality between the baseline and v4 model classes;
2. ordered-presentation equality against the frozen schedule;
3. a resolved-configuration whitelist admitting exactly one delta.

Evaluation-role scenes are never requested.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / ".generated/dev/DEVELOPMENT_ONLY_v4_occupied_weight_screen"
STATUS = "DEVELOPMENT_ONLY_NOT_CLAIM_BEARING"

V3_CONTRACT = (
    "lewm/benchmarks/go2_direct_egocentric_bev_signed_boundary_semantic_anchor_"
    "state_v3_update100_trend_gate_timing.py"
)
V3_AUTHORIZATION = (
    "docs/lewm_go2_rgb_direct_egocentric_bev_signed_boundary_semantic_anchor_"
    "state_v3_update100_trend_gate_timing_execution_authorization_2026-07-27.json"
)
LEAF_RUNNER = "scripts/run_go2_direct_egocentric_bev_state_jepa_v1.py"
V3_RUNNER = (
    "scripts/run_go2_direct_egocentric_bev_signed_boundary_semantic_anchor_"
    "state_v3_update100_trend_gate_timing.py"
)
V4_MODEL = "lewm/models/direct_egocentric_bev_signed_boundary_occupied_weight_v4.py"


def _module(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _state_sha(state: Any) -> str:
    digest = hashlib.sha256()
    for name in sorted(state):
        tensor = state[name].detach().to("cpu").contiguous()
        digest.update(name.encode())
        digest.update(str(tuple(tensor.shape)).encode())
        digest.update(str(tensor.dtype).encode())
        digest.update(tensor.numpy().tobytes(order="C"))
    return digest.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checks-only", action="store_true")
    ap.add_argument("--arm", choices=("A", "B"), required=True,
                    help="A = inherited macro-balanced O loss; B = committed v4 5:1")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    started = time.time()

    record: dict[str, Any] = {
        "status": STATUS,
        "claim_bearing": False,
        "passed_reviewed_execution_path": False,
        "note": (
            "the frozen v3 execution authorization was read only as a "
            "configuration source; it is not a grant for this run"
        ),
        "git": {
            "head": subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"],
                                   capture_output=True, text=True).stdout.strip(),
            "status_porcelain": subprocess.run(["git", "-C", str(ROOT), "status", "--porcelain"],
                                               capture_output=True, text=True).stdout,
        },
        "launcher_sha256": _sha(Path(__file__).resolve()),
        "v4_model_sha256": _sha(ROOT / V4_MODEL),
        "command": " ".join([sys.executable, *sys.argv]),
    }

    arm_out = OUT / f"arm_{args.arm}"
    arm_out.mkdir(parents=True, exist_ok=True)
    # The model chain reaches ``from lewm.models import ...`` partway through a
    # source-loaded import, so the package must be importable and the base
    # module fully initialised before the chain starts, or it observes a
    # partially initialised module.
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    import lewm.models.direct_egocentric_bev_state_jepa_v1 as _preload_v1  # noqa: F401

    contract = _module("_v4screen_contract", ROOT / V3_CONTRACT)
    # Use the COMPOSED v3 runner stack, not the bare leaf: the phase-schedule
    # arming lives in the v8 layer's _initialize_model override, and the bare
    # leaf raises "objective used before phase policy was armed".
    v3_runner = _module("_v4screen_v3_runner", ROOT / V3_RUNNER)
    v3_runner._rebind_inherited_runner()
    # The chain composes by monkey-patching a shared _LEAF rather than by
    # defining overrides, so the composed seam is reached through the anchor
    # runner.  Its _initialize_model is the v8 version, which arms the phase
    # schedule; the bare leaf's does not and raises "objective used before
    # phase policy was armed".
    leaf = v3_runner._V2._V1._LEAF
    if not hasattr(leaf, "_train_probe") or not hasattr(leaf, "_initialize_model"):
        raise RuntimeError("composed runner stack does not expose the training seam")
    import inspect as _inspect
    if "arm_phase_schedule_v6" not in _inspect.getsource(leaf._initialize_model):
        raise RuntimeError("composed _initialize_model does not arm the phase schedule")
    record["composed_initialize_model_defined_in"] = Path(
        _inspect.getfile(leaf._initialize_model)).name
    record["composed_runner_module"] = getattr(leaf, "__file__", "unknown")

    # DEVELOPMENT-ONLY: bind the CURRENT committed runtime rather than the
    # historical frozen manifest.  contract.current_source_bindings(ROOT) is the
    # claim-bearing closure validator and is deliberately NOT called here; this
    # screen is not on the reviewed execution path and inherits no manifest or
    # authorization status.  _load_post_reservation_stack still rehashes every
    # file against the bindings it is given, so the runtime remains internally
    # self-consistent.
    sources = {
        relative: _sha(ROOT / relative) for relative in contract.SOURCE_PATHS
    }
    record["runtime_source_bindings"] = {
        "count": len(sources),
        "bound_to": "current_HEAD_working_tree",
        "frozen_manifest_consulted": False,
        "sha256_of_bindings": hashlib.sha256(
            json.dumps(sources, sort_keys=True).encode()).hexdigest(),
    }
    matched, runtime, schedule_adapter, baseline_model_api = (
        leaf._load_post_reservation_stack(sources)
    )
    v4_model_api = _module("_v4screen_model", ROOT / V4_MODEL)
    arm_model_api = baseline_model_api if args.arm == "A" else v4_model_api
    record["arm"] = args.arm
    record["arm_objective"] = (
        "inherited macro-balanced O loss (~28:1 effective occupied:free)"
        if args.arm == "A" else
        "committed v4 fixed 5:1 occupied:free over known cells"
    )

    authorization = json.loads((ROOT / V3_AUTHORIZATION).read_text())
    runtime_inputs = authorization["runtime_inputs"]
    adapted = {"raw": runtime_inputs["raw"], "camera": runtime_inputs["n320"]}
    record["config_source"] = {
        "path": V3_AUTHORIZATION,
        "sha256": _sha(ROOT / V3_AUTHORIZATION),
        "used_as": "configuration_only_not_authorization",
    }

    progress: dict[str, Any] = {"stage": "dev_screen_inputs"}
    inputs = leaf._construct_raw_inputs_with_progress(matched, runtime, adapted, progress)
    leaf._normalize_endpoint_paths(inputs)
    train_pairs = inputs.role_pairs("train")
    selection_pairs = inputs.role_pairs("checkpoint_selection")
    train_mapping, selection_mapping, _perm = leaf._validate_target_mappings(
        train_pairs, selection_pairs
    )
    schedule, schedule_receipt = leaf._load_schedule(
        schedule_adapter, authorization, train_pairs, progress=progress
    )

    # ---- CHECK 3: resolved-configuration whitelist -------------------------
    whitelist = {
        "train_pairs": len(train_pairs) == contract.TRAIN_ROLE_COUNTS["pairs"],
        "selection_pairs": len(selection_pairs) == contract.SELECTION_ROLE_COUNTS["pairs"],
        "schedule_length": len(schedule) == contract.MAXIMUM_PRESENTATIONS,
        "roles_requested_are_train_and_selection_only": True,
        "evaluation_role_not_loaded": True,
    }
    record["resolved_config"] = {
        "train_pairs": len(train_pairs),
        "selection_pairs": len(selection_pairs),
        "schedule_length": len(schedule),
        "schedule_receipt": schedule_receipt,
        "corpus": adapted,
        "whitelist": whitelist,
    }

    # ---- CHECK 1: initial-state tensor equality ----------------------------
    device = runtime.torch.device("cuda")
    fit, n320_gate, n320_binding = leaf._load_n320_with_progress(
        matched, runtime, adapted, progress
    )
    record["n320_checkpoint_binding"] = n320_binding

    base_model, _p1, base_init = leaf._initialize_model(runtime, baseline_model_api, fit, device)
    v4_model, _p2, v4_init = leaf._initialize_model(runtime, v4_model_api, fit, device)
    base_sha = _state_sha(base_model.state_dict())
    v4_sha = _state_sha(v4_model.state_dict())
    record["initial_state"] = {
        "baseline_state_sha256": base_sha,
        "v4_state_sha256": v4_sha,
        "equal": base_sha == v4_sha,
    }

    # ---- CHECK 2: ordered-presentation equality ----------------------------
    schedule2, _r2 = leaf._load_schedule(
        schedule_adapter, authorization, train_pairs, progress=dict(progress)
    )
    ordered_equal = list(schedule) == list(schedule2)
    record["ordered_presentations"] = {
        "length": len(schedule),
        "sha256": hashlib.sha256(json.dumps(list(schedule)).encode()).hexdigest(),
        "deterministic_equal": ordered_equal,
    }

    checks_pass = (
        record["initial_state"]["equal"]
        and ordered_equal
        and all(whitelist.values())
    )
    record["fidelity_checks_passed"] = checks_pass
    if not checks_pass or args.checks_only:
        record["wall_seconds"] = time.time() - started
        (arm_out / "checks.json").write_text(json.dumps(record, indent=2, default=str))
        print(json.dumps({k: record[k] for k in
                          ("status", "fidelity_checks_passed", "initial_state",
                           "ordered_presentations", "resolved_config")},
                         indent=2, default=str))
        return 0 if checks_pass else 1

    # ---- Training: frozen implementation, v4 model_api only ---------------
    del base_model, v4_model
    # The training path writes snapshots through a custody-tracked output
    # registry that the reservation path normally initialises.  This is the same
    # single call the official runner makes (leaf line 1989), scoped to this
    # arm's development-only output root.
    leaf._reset_output_binding_registry(arm_out)
    record["output_registry_root"] = str(arm_out)

    loader = leaf.DirectBevNarrowLoader(runtime, inputs, progress=progress)
    progress["_loader"] = loader
    gpu_started = time.monotonic()
    (model, probe), determinism = leaf._run_with_strict_determinism(
        runtime,
        lambda: leaf._train_probe(
            runtime, arm_model_api, fit, loader, train_pairs, selection_pairs,
            train_mapping, selection_mapping, schedule, device, arm_out,
            gpu_started=gpu_started, progress=progress,
        ),
    )
    trained_class_module = type(model).__module__
    expected = "_v4screen_model" if args.arm == "B" else None
    record["trained_model_class"] = {
        "module": trained_class_module,
        "qualname": type(model).__qualname__,
        "arm": args.arm,
    }
    if args.arm == "B" and "_v4screen_model" not in trained_class_module:
        raise RuntimeError("arm B did not train the v4 model class")
    if args.arm == "A" and "_v4screen_model" in trained_class_module:
        raise RuntimeError("arm A trained the v4 model class")
    record["probe_status"] = str(probe.get("status"))
    record["determinism"] = determinism
    record["final_state_sha256"] = _state_sha(model.state_dict())
    record["wall_seconds"] = time.time() - started
    (arm_out / "result.json").write_text(json.dumps(record, indent=2, default=str))
    print(json.dumps({"status": STATUS, "probe_status": record["probe_status"],
                      "final_state_sha256": record["final_state_sha256"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
