#!/usr/bin/env python3
"""Continuation authorisation for registry seed indices 5, 6 and 7.

SCOPED_CONTINUATION_AUTHORISATION.

The receipt this module issues is deliberately shaped so that the EXISTING
verifier -- ``authorise_dev_proprio_launch_v1.verify`` -- accepts it unchanged.
That verifier already reads ``launch_commit`` and ``authorised_seed_indices``
from the receipt file rather than from its own constants, so continuing the
experiment requires **no edit to the trainer, the evaluator or any scientific
module**.  That is the point: the continuation adds authorisation machinery and
nothing else.

Before issuing, ``machine_check`` proves that every change since the original
launch commit falls inside the permitted categories and that no scientific
component moved.  The proof is embedded in the receipt.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import run_dev_proprio_factorial_driver_v1 as D  # noqa: E402
from scripts import build_dev_canonical_cache_map_v1 as MAP  # noqa: E402
from scripts import build_dev_factorial_manifest_v1 as FM  # noqa: E402
from scripts import freeze_dev_proprio_run_package_v1 as PKG  # noqa: E402
from scripts import authorise_dev_proprio_launch_v1 as AUTH  # noqa: E402

STATUS = "SCOPED_CONTINUATION_AUTHORISATION"
OUT = D.CACHE / "factorial_v1" / "continuation_authorisation.json"

PACKAGE_PROVENANCE_COMMIT = "d1d6440"
RUN_PACKAGE = "cf0456bef0cbe7cd8f2cd666b600f91ebf845f6156d180569edf36be53552991"
INITIAL_LAUNCH_COMMIT = "99a6eea"
INTERIM_COMMIT = "2f171bd"
INITIAL_LAUNCH_RECEIPT = "abe036ad3044467496ee1ead5cedef8ed40362220e841f23f7e443b45274a4fa"
VARIANCE_REPORT_DIGEST = "71d3dded6f53c204f5593a6c215cde8a89e164500a64f6fa3dd5d29c99219710"
FROZEN_N = 8
CONTINUATION_SEED_INDICES = (5, 6, 7)
LOCKED_SEED_INDICES = (8, 9)

# Files whose contents define the scientific computation.  None may differ
# between the original launch commit and the continuation launch commit.
SCIENTIFIC_MODULES = (
    "scripts/dev_proprio_predictor_v1.py",
    "scripts/run_dev_proprio_factorial_driver_v1.py",
    "scripts/eval_dev_proprio_seed_v1.py",
    "scripts/eval_dev_proprio_factorial_v1.py",
    "scripts/build_dev_canonical_cache_map_v1.py",
    "scripts/build_dev_factorial_manifest_v1.py",
    "scripts/build_dev_v03_proprio_action_manifest_v1.py",
    "scripts/dev_action_slew_reconstruction_v1.py",
    "scripts/dev_proprio_experiment_config_v1.py",
    "scripts/dev_seed_reestimation_v1.py",
    "scripts/freeze_dev_proprio_run_package_v1.py",
    "scripts/run_dev_v03_temporal_action_jepa_v1.py",
    "scripts/run_dev_v03_two_step_rollout_v1.py",
    "scripts/dev_checkpoint_v1.py",
    "scripts/authorise_dev_proprio_launch_v1.py",
)

# Every path permitted to differ, with the category that permits it.
PERMITTED_CHANGES = {
    "scripts/run_dev_proprio_variance_interim_v1.py":
        "the variance-only interim, including the structural blinding-guard correction",
    "scripts/continue_dev_proprio_launch_v1.py":
        "continuation authorisation machinery",
}
PERMITTED_PREFIXES = (("docs/", "documentation and reporting machinery"),)


class ContinuationRefused(RuntimeError):
    """The continuation may not be authorised."""


def _git(*args) -> str:
    return subprocess.run(["git", *args], cwd=ROOT, capture_output=True,
                          text=True).stdout.strip()


def machine_check() -> dict:
    """Prove the change set is confined and no scientific component moved."""
    changed = [line for line in
               _git("diff", "--name-only", f"{INITIAL_LAUNCH_COMMIT}..HEAD").splitlines()
               if line.strip()]
    disallowed, classified = [], {}
    for path in changed:
        if path in PERMITTED_CHANGES:
            classified[path] = PERMITTED_CHANGES[path]
            continue
        prefix = next((reason for start, reason in PERMITTED_PREFIXES
                       if path.startswith(start)), None)
        if prefix:
            classified[path] = prefix
            continue
        disallowed.append(path)

    scientific = {}
    for path in SCIENTIFIC_MODULES:
        diff = _git("diff", "--name-only", f"{INITIAL_LAUNCH_COMMIT}..HEAD", "--", path)
        scientific[path] = {
            "unchanged": diff == "",
            "blob_sha256": hashlib.sha256((ROOT / path).read_bytes()).hexdigest(),
        }
    moved = [path for path, entry in scientific.items() if not entry["unchanged"]]

    package = PKG.verify()
    factorial = FM.load()
    map_record = MAP.load()
    registry = D.register_seeds(D.CACHE / "factorial_v1")
    manifest = json.loads((D.PROPRIO / "proprio_manifest.json").read_text())

    digests = {
        "run_package": package["package_digest"] == RUN_PACKAGE,
        "factorial_manifest": factorial["digest"] == AUTH.AUTHORISED_FACTORIAL,
        "canonical_cache_map": map_record["digest"] == AUTH.AUTHORISED_MAP,
        "seed_registry": registry["sha256"] == AUTH.AUTHORISED_REGISTRY,
        "normalisation_contract": (manifest["normalisation_sha256"]
                                   == "f5ea58b29d79362d4d814ff1b4225b54a5c97fb95442c866def80b0c2c4c2fab"),
        "base_manifest_rows": (manifest["rows_sha256"]
                               == "7b79d12830f12175c591a87982a20e5df7a8d64cfc40e99dd9cee2dc1ae2543e"),
        "horizon_masks": (factorial["horizon_masks"]["mask_digest"]
                          == "ce32489f25a51b23c431dfe3591c0b7c571983ac8cbd5205d881ec86a32bfbfd"),
        "selection_rows_475": factorial["rows_by_split"]["checkpoint_selection"] == 475,
    }

    record = {
        "changed_files_since_initial_launch_commit": changed,
        "change_classification": classified,
        "disallowed_changes": disallowed,
        "all_changes_within_permitted_categories": not disallowed,
        "permitted_categories": [
            "attempt and result records (outside the repository, in the run cache)",
            "the variance-only interim",
            "launch or continuation authorisation",
            "the structural blinding-guard correction",
            "documentation and reporting machinery",
        ],
        "scientific_modules": scientific,
        "scientific_modules_moved": moved,
        "model_trainer_objectives_loaders_evaluator_unchanged": not moved,
        "package_bound_digests_match": digests,
        "all_package_bound_digests_match": all(digests.values()),
    }
    if disallowed:
        raise ContinuationRefused(
            f"changes outside the permitted categories: {disallowed}")
    if moved:
        raise ContinuationRefused(f"scientific modules changed: {moved}")
    if not all(digests.values()):
        raise ContinuationRefused(
            f"package-bound digests differ: "
            f"{[k for k, v in digests.items() if not v]}")
    return record


def build() -> dict:
    verification = machine_check()
    state = AUTH.git_state()
    if state["dirty"]:
        raise ContinuationRefused(
            f"cannot issue a continuation receipt from a dirty tree:\n{state['porcelain']}")
    registry = D.register_seeds(D.CACHE / "factorial_v1")
    interim = json.loads(
        (D.CACHE / "factorial_v1" / "variance_only_interim.json").read_text())
    if interim["report_digest"] != VARIANCE_REPORT_DIGEST:
        raise ContinuationRefused("variance-only interim digest differs")
    if interim["frozen_total_N"] != FROZEN_N:
        raise ContinuationRefused(
            f"interim froze N={interim['frozen_total_N']}, not {FROZEN_N}")

    receipt = {
        "status": STATUS,
        "scope": ("continuation of the frozen experiment: registry seed indices 5, 6 and 7 "
                  "only, to reach the frozen total of eight quadruplets"),
        # provenance, kept distinct
        "package_provenance_commit": PACKAGE_PROVENANCE_COMMIT,
        "initial_launch_commit": INITIAL_LAUNCH_COMMIT,
        "completed_interim_commit": INTERIM_COMMIT,
        "launch_commit": state["head"],
        "launch_commit_short": state["short"],
        "continuation_launch_commit_contains": (
            "only authorisation and reporting machinery; no scientific module differs "
            "from the initial launch commit"),
        # bound artefacts
        "run_package_digest": RUN_PACKAGE,
        "factorial_manifest_digest": AUTH.AUTHORISED_FACTORIAL,
        "canonical_map_digest": AUTH.AUTHORISED_MAP,
        "seed_registry_digest": AUTH.AUTHORISED_REGISTRY,
        "initial_launch_receipt_digest": INITIAL_LAUNCH_RECEIPT,
        "variance_only_report_digest": VARIANCE_REPORT_DIGEST,
        # frozen design
        "frozen_total_N": FROZEN_N,
        "sample_size_recalculation": "PROHIBITED -- the frozen N is final",
        "authorised_seed_indices": list(CONTINUATION_SEED_INDICES),
        "authorised_seeds": [D.SEED_REGISTRY[i] for i in CONTINUATION_SEED_INDICES],
        "locked_seed_indices": list(LOCKED_SEED_INDICES),
        "locked_seeds": [D.SEED_REGISTRY[i] for i in LOCKED_SEED_INDICES],
        "locked_seeds_note": ("indices 8 and 9 are NOT authorised as substitutes or "
                              "optional extensions"),
        "cell_order": {str(i): list(D.cell_order(i)) for i in CONTINUATION_SEED_INDICES},
        "device_policy": {"device_index": AUTH.AUTHORISED_DEVICE_INDEX,
                          "device_name": AUTH.AUTHORISED_DEVICE,
                          "one_cell_at_a_time": True},
        "prohibited": [
            "rerunning, extending, omitting or replacing a finite run because of its "
            "performance",
            "recalculating or revising the sample size",
            "running another interim after seed 5, 6 or 7",
            "launching registry indices 8 or 9",
        ],
        "blinding": ("operational and technical-validity information only until all eight "
                     "quadruplets are complete"),
        "machine_check": verification,
        "registry_sha256_at_issue": registry["sha256"],
    }
    receipt["receipt_digest"] = hashlib.sha256(
        json.dumps(receipt, sort_keys=True).encode()).hexdigest()
    return receipt


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--check-only", action="store_true")
    args = ap.parse_args()
    if args.check_only:
        print(json.dumps(machine_check(), indent=2)[:4000])
        return 0
    receipt = build()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(receipt, indent=2))
    print(json.dumps({k: v for k, v in receipt.items() if k != "machine_check"}, indent=2))
    print("\nmachine_check summary:", json.dumps({
        k: receipt["machine_check"][k] for k in
        ("changed_files_since_initial_launch_commit",
         "all_changes_within_permitted_categories",
         "model_trainer_objectives_loaders_evaluator_unchanged",
         "all_package_bound_digests_match")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
