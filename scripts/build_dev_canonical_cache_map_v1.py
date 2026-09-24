#!/usr/bin/env python3
"""The one canonical manifest-to-cache map, shared by trainer and evaluator.

DEVELOPMENT_ONLY_NOT_CLAIM_BEARING.

Row identity is **structural**, never the pair hash alone:

    (scene, env_index, episode_id, reset_count,
     source_frame_index, target_frame_index)

The pair content hash is then used to VERIFY that the row resolved structurally
carries the visual contents it should.  Identity and verification are separate
jobs and are kept separate: a hash collision or a re-hash would otherwise silently
re-point a row, and a structural collision would silently be resolved by first
match.  If a structural identifier is non-unique this module STOPS and reports it.

The cached feature blobs are indexed by position within the ORIGINAL 4,566-row
``temporal_rows.jsonl`` (train rows in order, then selection rows in order).  The
retained manifest is a 4,444-row subset, so a filtered loader must never reuse the
original position: it must go through ``cache_index`` recorded here.
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

STATUS = "DEVELOPMENT_ONLY_NOT_CLAIM_BEARING"

CACHE = Path("/home/andrewknowles/.cache/lewm_go2_temporal_v03")
SOURCE_ROWS = CACHE / "temporal_rows.jsonl"
TWO_STEP_ROWS = CACHE / "two_step" / "two_step_rows.jsonl"
PROPRIO = CACHE / "proprio_v1"
OUT = PROPRIO / "canonical_cache_map.json"

FRAMES_PER_TIMESTEP = 48
TARGET_OFFSET = 240


class MapViolation(RuntimeError):
    """A structural defect: the map must not be written."""


def structural_key(scene, env_index, episode_id, reset_count, source, target):
    return (str(scene), int(env_index), int(episode_id), int(reset_count),
            int(source), int(target))


def stable_row_id(key) -> str:
    """A short, stable identifier derived from the structural fields only."""
    material = "|".join(str(part) for part in key).encode()
    return hashlib.sha256(material).hexdigest()[:32]


def episode_cluster(scene, env_index, episode_id, reset_count) -> str:
    return f"{scene}/env{int(env_index):02d}/ep{int(episode_id)}/r{int(reset_count)}"


def build() -> dict:
    source = [json.loads(line) for line in SOURCE_ROWS.read_text().splitlines() if line.strip()]
    train = [r for r in source if r["role"] == "train"]
    selection = [r for r in source if r["role"] == "checkpoint_selection"]

    # A THIRD index space: the step-2 target blobs are indexed by position within
    # two_step_rows.jsonl, not by the base-row position.  A rollout cell needs a
    # step-2 target, so this index has to travel in the same map or the two
    # objectives would silently see different rows.
    two_step = [json.loads(line) for line in TWO_STEP_ROWS.read_text().splitlines()
                if line.strip()]
    step2_position = {}
    for role in ("train", "checkpoint_selection"):
        for index, row in enumerate([r for r in two_step if r["role"] == role]):
            key = (role, row["pair_sha256"])
            if key in step2_position:
                raise MapViolation(f"two-step rows carry a duplicate pair hash: {key}")
            step2_position[key] = index

    # ---- structural index over the ORIGINAL rows, with a uniqueness gate -----
    by_key, duplicates = {}, collections.defaultdict(list)
    for role, bucket in (("train", train), ("checkpoint_selection", selection)):
        for cache_index, row in enumerate(bucket):
            frames = {f["offset"]: f["frame_index"] for f in row["frames"]}
            key = structural_key(row["scene"], row["env_index"], row["episode_id"],
                                 row["reset_count"], frames[0], frames[TARGET_OFFSET])
            if key in by_key:
                duplicates[key].append((role, cache_index))
            else:
                by_key[key] = {"role": role, "cache_index": cache_index, "row": row,
                               "source_frame": frames[0],
                               "target_frame": frames[TARGET_OFFSET]}
    if duplicates:
        raise MapViolation(
            f"structural identifier is NOT unique for {len(duplicates)} key(s); refusing to "
            f"resolve by first match. Example: {next(iter(duplicates.items()))}")

    # ---- resolve every retained manifest row --------------------------------
    manifest = json.loads((PROPRIO / "proprio_manifest.json").read_text())
    retained = [json.loads(line) for line in
                (PROPRIO / "proprio_rows.jsonl").read_text().splitlines() if line.strip()]

    entries, used = [], {}
    for manifest_row_index, row in enumerate(retained):
        source_frame = row["t"]
        key = structural_key(row["scene"], row["env_index"], row["episode_id"],
                             row["reset_count"], source_frame, source_frame + TARGET_OFFSET)
        found = by_key.get(key)
        if found is None:
            raise MapViolation(f"manifest row {manifest_row_index} resolves to no cache row: {key}")
        slot = (found["role"], found["cache_index"])
        if slot in used:
            raise MapViolation(
                f"cache row {slot} claimed by manifest rows {used[slot]} and {manifest_row_index}")
        used[slot] = manifest_row_index

        original = found["row"]
        if original["pair_sha256"] != row["pair_sha256"]:
            raise MapViolation(
                f"row {manifest_row_index}: pair hash disagrees "
                f"({original['pair_sha256'][:12]} vs {row['pair_sha256'][:12]})")
        for field in ("scene", "family", "env_index", "episode_id", "reset_count", "role"):
            if original[field] != row[field]:
                raise MapViolation(
                    f"row {manifest_row_index}: structural metadata '{field}' disagrees "
                    f"({original[field]!r} vs {row[field]!r})")
        if (source_frame % FRAMES_PER_TIMESTEP) != row["env_index"]:
            raise MapViolation(
                f"row {manifest_row_index}: frame index {source_frame} is inconsistent with "
                f"env {row['env_index']}")

        entries.append({
            "manifest_row_index": manifest_row_index,
            "stable_row_id": stable_row_id(key),
            "cache_index": found["cache_index"],
            "split": row["role"],
            "family": row["family"],
            "episode_cluster": episode_cluster(row["scene"], row["env_index"],
                                               row["episode_id"], row["reset_count"]),
            "source_frame_index": source_frame,
            "target_frame_index": source_frame + TARGET_OFFSET,
            "pair_sha256": row["pair_sha256"],
            "has_action": bool(row["action_blocks"]),
            "action_blocks_available": len(row["action_blocks"]),
            "has_control": bool(row["control"]) and len(row["control"]) == 15,
            "has_proprio": bool(row["proprio"]) and len(row["proprio"]) == 15,
            "step2_cache_index": step2_position.get((row["role"], row["pair_sha256"])),
            "has_step2_target": (row["role"], row["pair_sha256"]) in step2_position,
        })

    # ---- exclusions must be accounted for -----------------------------------
    retained_keys = {structural_key(r["scene"], r["env_index"], r["episode_id"],
                                    r["reset_count"], r["t"], r["t"] + TARGET_OFFSET)
                     for r in retained}
    excluded = [key for key in by_key if key not in retained_keys]
    dropped_total = sum(manifest["rows_dropped"].values())
    if len(excluded) != dropped_total:
        raise MapViolation(
            f"{len(excluded)} rows absent from the manifest but the drop ledger records "
            f"{dropped_total}")

    # ---- splits must be disjoint --------------------------------------------
    train_ids = {e["stable_row_id"] for e in entries if e["split"] == "train"}
    selection_ids = {e["stable_row_id"] for e in entries if e["split"] == "checkpoint_selection"}
    if train_ids & selection_ids:
        raise MapViolation(f"{len(train_ids & selection_ids)} row(s) in both splits")
    if len({e["stable_row_id"] for e in entries}) != len(entries):
        raise MapViolation("stable row identifiers are not unique")

    with_step2 = [e for e in entries if e["has_step2_target"]]
    per_split = collections.Counter(e["split"] for e in entries)
    per_family = collections.Counter(f"{e['split']}/{e['family']}" for e in entries)

    record = {
        "status": STATUS, "claim_bearing": False,
        "identity": ("structural: (scene, env_index, episode_id, reset_count, "
                     "source_frame_index, target_frame_index); the pair hash VERIFIES "
                     "contents and is never the identity"),
        "cache_indexing": ("position within the ORIGINAL 4,566-row temporal_rows.jsonl, "
                           "train rows then selection rows; a filtered loader must use "
                           "cache_index and never a post-filter position"),
        "source_rows": len(source), "source_train": len(train), "source_selection": len(selection),
        "retained_rows": len(entries),
        "retained_by_split": dict(per_split),
        "retained_by_split_and_family": dict(per_family),
        "excluded_rows": len(excluded),
        "drop_ledger_total": dropped_total,
        "episode_clusters": len({e["episode_cluster"] for e in entries}),
        "rows_with_step2_target": len(with_step2),
        "rows_with_step2_by_split": dict(collections.Counter(e["split"] for e in with_step2)),
        "factorial_row_set": ("rows carrying a step-2 target: the rollout objective needs one, "
                              "and all four cells must see identical rows, so the one-step "
                              "cells are restricted to the same set"),
        "step2_cache_indexing": ("position within two_step_rows.jsonl per split -- a THIRD "
                                 "index space, distinct from the base-row position"),
        "manifest_rows_sha256": manifest["rows_sha256"],
        "normalisation_sha256": manifest["normalisation_sha256"],
        "verification": {
            "every_manifest_row_resolves_to_exactly_one_cache_row": True,
            "no_cache_row_claimed_twice": True,
            "structural_metadata_agrees": True,
            "pair_hashes_agree": True,
            "excluded_rows_accounted_for_in_drop_ledger": True,
            "train_and_selection_indices_disjoint": True,
            "structural_identifier_unique": True,
        },
        "entries": entries,
    }
    payload = json.dumps({k: v for k, v in record.items() if k != "digest"}, sort_keys=True)
    record["digest"] = hashlib.sha256(payload.encode()).hexdigest()
    return record


def load(path: Path = OUT) -> dict:
    record = json.loads(Path(path).read_text())
    stored = record.pop("digest")
    recomputed = hashlib.sha256(
        json.dumps(record, sort_keys=True).encode()).hexdigest()
    if recomputed != stored:
        raise MapViolation(f"canonical map digest mismatch: {recomputed} != {stored}")
    record["digest"] = stored
    return record


def cache_indices(record: dict, split: str):
    """Ordered (manifest_row_index, cache_index) for one split -- the ONLY lookup path."""
    return [(e["manifest_row_index"], e["cache_index"])
            for e in record["entries"] if e["split"] == split]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()
    record = build()
    Path(args.out).write_text(json.dumps(record, indent=2))
    summary = {k: v for k, v in record.items() if k != "entries"}
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
