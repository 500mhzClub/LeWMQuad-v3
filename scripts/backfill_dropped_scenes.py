#!/usr/bin/env python3
"""Propose salt-rerolled replacements for corpus scenes dropped at rollout.

A scene can pass the planner's reachability gate yet blow up the Genesis rigid
solver at rollout (``Invalid constraint forces causing 'nan'``). Those scenes
are skipped during datagen, leaving the *output* short of the planned per-family
count. This tool finds, for each dropped ``scene_id``, the next deterministic
salt (past the value the planner chose) whose seed is collision-free and
reachability-valid — producing an alternative ``scene_id`` for the same plan
slot.

This module only *proposes* (and, with the future --apply path, realizes) seeds.
The NaN re-test itself is driven by the caller, which rolls out each proposed
candidate in an isolated corpus and advances the salt if it still NaNs.

Usage (dry-run):
    PYTHONPATH=lewm_worlds:lewm_genesis python3 scripts/backfill_dropped_scenes.py \
        --corpus <corpus_dir> --dropped <scene_id> [--dropped <scene_id> ...]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import tempfile
from pathlib import Path

from lewm_worlds.splits import (
    _scene_seed, _default_scene_validator, _MAX_VALIDATION_SALT,
    CorpusPlan, SceneAssignment, PLAN_VERSION,
)
from lewm_worlds.manifest import stable_scene_id
from lewm_worlds.families import build_family_manifest
from lewm_worlds.corpus import build_corpus

REPO = Path(__file__).resolve().parent.parent


def _plan_sha256_dict(plan_dict: dict) -> str:
    # Mirror lewm_worlds.splits.plan_sha256 exactly (compact, sorted).
    payload = json.dumps(plan_dict, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _rollout_survives(temp_corpus: Path, split: str, family: str, scene_id: str,
                      n_envs: int, n_blocks: int) -> tuple[bool, Path]:
    """Roll out the single scene in ``temp_corpus`` end-to-end; return
    (survived, temp_out). 'Survived' = it produced labels (not NaN-skipped)."""
    temp_out = temp_corpus.parent / "out"
    cmd = [
        "bash", str(REPO / "scripts" / "run_mass_datagen.sh"),
        "--scene-corpus", str(temp_corpus),
        "--split", split, "--family", family, "--scene-limit", "1",
        "--n-envs", str(n_envs), "--n-blocks", str(n_blocks),
        "--backend", "cpu", "--no-render", "--out", str(temp_out),
    ]
    log = temp_corpus.parent / "rollout.log"
    with open(log, "w") as fh:
        rc = subprocess.run(cmd, stdout=fh, stderr=subprocess.STDOUT,
                            env={**_env(), "JOBS": "auto"}).returncode
    survived = (temp_out / "labels" / scene_id).is_dir() and rc == 0
    return survived, temp_out


def _env() -> dict:
    import os
    e = dict(os.environ)
    pp = e.get("PYTHONPATH", "")
    e["PYTHONPATH"] = f"{REPO/'lewm_genesis'}:{REPO/'lewm_worlds'}" + (f":{pp}" if pp else "")
    return e


def _build_one_scene(dest_corpus: Path, plan_seed: int, split: str, family: str,
                     index: int, seed: int, salt: int, scene_id: str) -> dict:
    """Write a single-scene corpus at ``dest_corpus`` and return its scene
    result dict (from the emitted corpus.json 'scenes' list)."""
    a = SceneAssignment(split=split, family=family, scene_index=index,
                        scene_seed=seed, scene_id=scene_id, scene_seed_salt=salt)
    plan1 = CorpusPlan(plan_seed=plan_seed, plan_version=PLAN_VERSION,
                       splits=(split,), families=(family,),
                       totals={split: {family: 1}}, assignments=(a,))
    build_corpus(plan1, dest_corpus, emit_genesis=True)
    scenes = json.loads((dest_corpus / "corpus.json").read_text())["scenes"]
    return next(s for s in scenes if s["scene_id"] == scene_id)


def next_valid_salt(plan_seed, split, family, index, start_salt, seen_seeds, max_candidates=1):
    """Yield up to ``max_candidates`` (salt, seed, scene_id) past ``start_salt``
    whose seed is collision-free and reachability-valid (same gate the planner
    used)."""
    found = []
    for salt in range(start_salt, _MAX_VALIDATION_SALT + 1):
        seed = _scene_seed(plan_seed=plan_seed, split=split, family=family, index=index, salt=salt)
        if seed in seen_seeds:
            continue
        manifest = build_family_manifest(
            scene_seed=seed, family=family, split=split, difficulty_tier=None
        )
        if _default_scene_validator(manifest):
            found.append((salt, seed, stable_scene_id(family, seed)))
            if len(found) >= max_candidates:
                break
    return found


def _resolve_surviving(plan_seed, split, fam, idx, start_salt, seen, n_envs, n_blocks, workdir):
    """Iterate salts past ``start_salt``; for each reachability-valid,
    collision-free seed, build+roll out the scene; return the first that
    survives (no NaN), else None."""
    for salt in range(start_salt, _MAX_VALIDATION_SALT + 1):
        seed = _scene_seed(plan_seed=plan_seed, split=split, family=fam, index=idx, salt=salt)
        if seed in seen:
            continue
        manifest = build_family_manifest(scene_seed=seed, family=fam, split=split,
                                         difficulty_tier=None)
        if not _default_scene_validator(manifest):
            continue
        new_id = stable_scene_id(fam, seed)
        td = Path(tempfile.mkdtemp(prefix=f"bf_{new_id}_", dir=workdir))
        temp_corpus = td / "corpus"
        scene_result = _build_one_scene(temp_corpus, plan_seed, split, fam, idx,
                                        seed, salt, new_id)
        print(f"   salt={salt} -> {new_id}: rolling out (1 scene)...", flush=True)
        survived, temp_out = _rollout_survives(temp_corpus, split, fam, new_id, n_envs, n_blocks)
        if survived:
            print(f"   salt={salt} -> {new_id}: SURVIVED")
            return dict(salt=salt, seed=seed, new_id=new_id, scene_result=scene_result,
                        temp_corpus=temp_corpus, temp_out=temp_out, old_id=None,
                        split=split, fam=fam, idx=idx)
        print(f"   salt={salt} -> {new_id}: NaN/failed (see {td/'rollout.log'}), advancing")
        shutil.rmtree(td, ignore_errors=True)
    return None


def run_dry(args, corpus, plan_seed, by_id, seen_seeds):
    for sid in args.dropped:
        a = by_id.get(sid)
        if a is None:
            print(f"!! {sid}: NOT FOUND in plan assignments")
            continue
        split, fam, idx = a["split"], a["family"], a["scene_index"]
        ok = stable_scene_id(fam, a["scene_seed"]) == sid
        print(f"== {sid}\n   slot: split={split} family={fam} index={idx}")
        print(f"   current: salt={a['scene_seed_salt']} seed={a['scene_seed']} id-matches-seed={ok}")
        cands = next_valid_salt(plan_seed, split, fam, idx, a["scene_seed_salt"] + 1,
                                seen_seeds, max_candidates=args.candidates)
        if not cands:
            print("   !! no reachability-valid reroll within salt budget")
            continue
        for n, (salt, seed, new_id) in enumerate(cands):
            print(f"   candidate salt={salt:<3} seed={seed:<22} -> {new_id}"
                  + ("  <- would use" if n == 0 else ""))
        print("")
    print("DRY-RUN ONLY — no scenes built, no rollout, corpus.json untouched.")
    return 0


def run_apply(args, corpus, plan_seed, by_id, seen_seeds):
    corpus_dir = args.corpus
    out_root = args.out_root
    seen = set(seen_seeds)
    workdir = Path(tempfile.mkdtemp(prefix="backfill_work_", dir=out_root))
    swaps = []
    for sid in args.dropped:
        a = by_id.get(sid)
        if a is None:
            print(f"!! {sid}: NOT FOUND in plan — ABORT, no changes made"); return 1
        print(f"== resolving replacement for {sid} "
              f"(slot {a['split']}/{a['family']}#{a['scene_index']})", flush=True)
        res = _resolve_surviving(plan_seed, a["split"], a["family"], a["scene_index"],
                                 a["scene_seed_salt"] + 1, seen, args.n_envs, args.n_blocks, workdir)
        if res is None:
            print(f"!! no surviving replacement for {sid} within salt budget — "
                  f"ABORT, no corpus changes made"); return 1
        res["old_id"] = sid
        seen.add(res["seed"])
        swaps.append(res)

    # All resolved — promote and patch atomically at the end.
    for s in swaps:
        chunk = out_root / "rollout" / s["split"] / s["fam"] / "chunk_backfill"
        for sub in ("rollout", "raw", "labels"):
            (chunk / sub).mkdir(parents=True, exist_ok=True)
        # scene definition into the real corpus
        src_def = s["temp_corpus"] / s["split"] / s["fam"] / s["new_id"]
        dst_def = corpus_dir / s["split"] / s["fam"] / s["new_id"]
        shutil.rmtree(dst_def, ignore_errors=True)
        shutil.copytree(src_def, dst_def)
        # output (rollout/raw/labels) into chunk_backfill
        for sub in ("rollout", "raw", "labels"):
            dst = chunk / sub / s["new_id"]
            shutil.rmtree(dst, ignore_errors=True)
            shutil.move(str(s["temp_out"] / sub / s["new_id"]), str(dst))
        # patch plan assignment in place (keep split/family/scene_index)
        for asg in corpus["plan"]["assignments"]:
            if asg["scene_id"] == s["old_id"]:
                asg["scene_seed"] = s["seed"]
                asg["scene_id"] = s["new_id"]
                asg["scene_seed_salt"] = s["salt"]
                break
        # patch scenes list: drop old, add new
        corpus["scenes"] = [x for x in corpus["scenes"] if x["scene_id"] != s["old_id"]]
        corpus["scenes"].append(s["scene_result"])
        # remove the old NaN-prone scene definition
        shutil.rmtree(corpus_dir / s["split"] / s["fam"] / s["old_id"], ignore_errors=True)
        (chunk / ".chunk_done").touch()
        print(f"   promoted {s['old_id']} -> {s['new_id']} (salt={s['salt']}) into {chunk}")

    corpus["plan_sha256"] = _plan_sha256_dict(corpus["plan"])
    (corpus_dir / "corpus.json").write_text(
        json.dumps(corpus, indent=2, sort_keys=True), encoding="utf-8")
    shutil.rmtree(workdir, ignore_errors=True)
    print(f"new plan_sha256 = {corpus['plan_sha256']}")
    print(f"BACKFILL_DONE swaps={len(swaps)}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", type=Path, required=True)
    ap.add_argument("--dropped", action="append", default=[], required=True,
                    help="scene_id that was dropped at rollout (repeatable)")
    ap.add_argument("--candidates", type=int, default=3,
                    help="how many valid alternatives to list per slot (dry-run)")
    ap.add_argument("--apply", action="store_true",
                    help="actually roll out replacements and patch corpus.json")
    ap.add_argument("--out-root", type=Path, default=None,
                    help="datagen output root (required with --apply)")
    ap.add_argument("--n-envs", type=int, default=48)
    ap.add_argument("--n-blocks", type=int, default=200)
    args = ap.parse_args()

    corpus = json.loads((args.corpus / "corpus.json").read_text())
    plan = corpus["plan"]
    plan_seed = plan["plan_seed"]
    by_id = {a["scene_id"]: a for a in plan["assignments"]}
    seen_seeds = {a["scene_seed"] for a in plan["assignments"]}

    print(f"corpus      = {args.corpus}")
    print(f"plan_seed   = {plan_seed}   plan_version = {plan.get('plan_version')}")
    print(f"assignments = {len(plan['assignments'])}   max_validation_salt = {_MAX_VALIDATION_SALT}")
    print("")

    if not args.apply:
        return run_dry(args, corpus, plan_seed, by_id, seen_seeds)
    if args.out_root is None:
        ap.error("--apply requires --out-root")
    return run_apply(args, corpus, plan_seed, by_id, seen_seeds)


if __name__ == "__main__":
    raise SystemExit(main())
