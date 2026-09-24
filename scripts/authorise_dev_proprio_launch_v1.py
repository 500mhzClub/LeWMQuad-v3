#!/usr/bin/env python3
"""Scoped launch authorisation for the initial five-seed stage.

DEVELOPMENT_ONLY_NOT_CLAIM_BEARING as an artefact; it authorises a scientific
training stage but is not itself a claim.

The receipt binds a source commit, four artefact digests, the first five seed
entries and the device policy.  ``verify`` re-checks every binding at launch and
REFUSES if the source tree is dirty, any bound digest differs, the GPU identity
differs, or a requested seed lies outside the authorised prefix.

This module authorises; it changes nothing.  No configuration, model behaviour,
dataset artefact, normalisation statistic, metric definition, checkpoint rule,
seed value or execution order is touched here or anywhere downstream of here.
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

STATUS = "SCOPED_LAUNCH_AUTHORISATION"
OUT = D.CACHE / "factorial_v1" / "launch_authorisation.json"

# The authorised SOURCE COMMIT is the provenance of the approved run package.
# The launch harness (this module, the driver's scientific path, the per-seed
# evaluator) cannot exist at that commit -- it is what launches it -- so the
# receipt binds two commits:
#
#   package_provenance_commit  d1d6440, the authorised scientific state; every
#                              scientific digest is re-verified to be UNCHANGED
#   launch_commit              the immutable source state the stage runs from,
#                              pinned when the receipt is issued
#
# Nothing scientific may differ between them, and HEAD may not drift from
# launch_commit for the duration of the stage.  Both are checked at every launch.
AUTHORISED_COMMIT = "d1d6440"
AUTHORISED_PACKAGE = "cf0456bef0cbe7cd8f2cd666b600f91ebf845f6156d180569edf36be53552991"
AUTHORISED_FACTORIAL = "6ff053033475debd3d8bb415080efb15adfaefc31f01295b956bd85c12b6dac0"
AUTHORISED_MAP = "a45bcc7d46da3c085f0603e79e568f1228b76c489868d6a96aed2b1485d85a7e"
AUTHORISED_REGISTRY = "bbaffee2f246813778e7c7195794414541dc9d298b6877df8562f359f21ba3a6"
AUTHORISED_SEED_COUNT = 5
AUTHORISED_DEVICE = "AMD Radeon AI PRO R9700"
AUTHORISED_DEVICE_INDEX = 0


class LaunchRefused(RuntimeError):
    """Any binding failed: no scientific run may start."""


def git_state() -> dict:
    def run(*args):
        return subprocess.run(["git", *args], cwd=ROOT, capture_output=True,
                              text=True).stdout.strip()
    return {"head": run("rev-parse", "HEAD"),
            "short": run("rev-parse", "--short", "HEAD"),
            "dirty": run("status", "--porcelain") != "",
            "porcelain": run("status", "--porcelain")}


def build() -> dict:
    registry = D.register_seeds(D.CACHE / "factorial_v1")
    receipt = {
        "status": STATUS,
        "scope": "initial scientific stage: the first five seed quadruplets only",
        "package_provenance_commit": AUTHORISED_COMMIT,
        "launch_commit": git_state()["head"],
        "launch_commit_short": git_state()["short"],
        "scientific_digests_unchanged_since_provenance": True,
        "run_package_digest": AUTHORISED_PACKAGE,
        "factorial_manifest_digest": AUTHORISED_FACTORIAL,
        "canonical_map_digest": AUTHORISED_MAP,
        "seed_registry_digest": AUTHORISED_REGISTRY,
        "authorised_seeds": list(D.SEED_REGISTRY[:AUTHORISED_SEED_COUNT]),
        "authorised_seed_indices": list(range(AUTHORISED_SEED_COUNT)),
        "locked_seeds": list(D.SEED_REGISTRY[AUTHORISED_SEED_COUNT:]),
        "cell_order": {str(i): list(D.cell_order(i)) for i in range(AUTHORISED_SEED_COUNT)},
        "device_policy": {"device_index": AUTHORISED_DEVICE_INDEX,
                          "device_name": AUTHORISED_DEVICE,
                          "one_cell_at_a_time": True},
        "prohibited": [
            "altering any scientific configuration, model behaviour, dataset artefact, "
            "normalisation statistic, metric definition, checkpoint rule, seed value or "
            "execution order",
            "best-checkpoint selection, training extension, omitting a poor run, "
            "reweighting, or repeating a valid run because of its metrics",
            "launching seeds six to ten",
        ],
        "registry_sha256_at_issue": registry["sha256"],
    }
    receipt["receipt_digest"] = hashlib.sha256(
        json.dumps(receipt, sort_keys=True).encode()).hexdigest()
    return receipt


def verify(seed_index: int | None = None, path: Path = OUT, require_clean=True) -> dict:
    """Re-check every binding.  Raises LaunchRefused on any mismatch."""
    receipt = json.loads(Path(path).read_text())
    stored = receipt.pop("receipt_digest")
    if hashlib.sha256(json.dumps(receipt, sort_keys=True).encode()).hexdigest() != stored:
        raise LaunchRefused("launch receipt digest mismatch")
    receipt["receipt_digest"] = stored

    failures = []
    state = git_state()
    if require_clean and state["dirty"]:
        failures.append(f"source tree is dirty:\n{state['porcelain']}")
    if state["head"] != receipt["launch_commit"]:
        failures.append(
            f"HEAD {state['short']} != the pinned launch commit "
            f"{receipt['launch_commit_short']}; the source may not change during the stage")
    provenance = subprocess.run(
        ["git", "merge-base", "--is-ancestor", receipt["package_provenance_commit"], "HEAD"],
        cwd=ROOT, capture_output=True)
    if provenance.returncode != 0:
        failures.append(
            f"the authorised package-provenance commit "
            f"{receipt['package_provenance_commit']} is not an ancestor of HEAD")

    package = PKG.verify()
    if package["package_digest"] != receipt["run_package_digest"]:
        failures.append("run package digest differs")
    factorial = FM.load()
    if factorial["digest"] != receipt["factorial_manifest_digest"]:
        failures.append("factorial manifest digest differs")
    map_record = MAP.load()
    if map_record["digest"] != receipt["canonical_map_digest"]:
        failures.append("canonical map digest differs")
    registry = D.register_seeds(D.CACHE / "factorial_v1")
    if registry["sha256"] != receipt["seed_registry_digest"]:
        failures.append("seed registry digest differs")

    try:
        import torch
        if torch.cuda.is_available():
            index = receipt["device_policy"]["device_index"]
            name = torch.cuda.get_device_name(index)
            if name != receipt["device_policy"]["device_name"]:
                failures.append(f"device {index} is {name!r}, not the authorised GPU")
        else:
            failures.append("no CUDA/ROCm device available")
    except Exception as error:                                  # pragma: no cover
        failures.append(f"device check failed: {error}")

    if seed_index is not None and seed_index not in receipt["authorised_seed_indices"]:
        failures.append(
            f"seed index {seed_index} lies outside the authorised prefix "
            f"{receipt['authorised_seed_indices']}")

    if failures:
        raise LaunchRefused("launch refused:\n  - " + "\n  - ".join(failures))
    return receipt


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--verify", action="store_true")
    ap.add_argument("--seed-index", type=int, default=None)
    ap.add_argument("--allow-dirty", action="store_true",
                    help="issue the receipt before committing it; launch still refuses")
    args = ap.parse_args()
    path = Path(args.out)
    if args.verify:
        receipt = verify(args.seed_index, path, require_clean=not args.allow_dirty)
        print(json.dumps({"verified": True, "receipt_digest": receipt["receipt_digest"],
                          "authorised_seeds": receipt["authorised_seeds"]}, indent=2))
        return 0
    receipt = build()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(receipt, indent=2))
    print(json.dumps(receipt, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
