"""Deterministic corpus planning with split- and family-disjoint seeds.

The planner converts a desired per-split, per-family scene count into a flat,
reproducible list of ``(split, family, scene_seed)`` assignments. Seeds are
derived by hashing ``(plan_seed, split, family, index)``, so two splits never
share a topology seed even if their family counts overlap.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from typing import Any

from lewm_worlds.families import registered_families
from lewm_worlds.manifest import stable_scene_id


# Splits supported out of the box. Callers may extend ``totals`` with custom
# splits — the planner is split-name agnostic.
DEFAULT_SPLITS: tuple[str, ...] = ("train", "val", "test_id", "test_hard")


# Shares (data spec section 8) used by ``smoke_corpus_plan`` and
# ``standard_corpus_plan`` below. Hard test deliberately reweights toward
# large/alias families.
TRAIN_SHARES: dict[str, float] = {
    "open_obstacle_field": 0.15,
    "small_enclosed_maze": 0.20,
    "medium_enclosed_maze": 0.35,
    "large_enclosed_maze": 0.20,
    "loop_alias_stress": 0.10,
}

HARD_TEST_SHARES: dict[str, float] = {
    "open_obstacle_field": 0.05,
    "small_enclosed_maze": 0.10,
    "medium_enclosed_maze": 0.25,
    "large_enclosed_maze": 0.35,
    "loop_alias_stress": 0.25,
}


@dataclass(frozen=True)
class SceneAssignment:
    split: str
    family: str
    scene_index: int
    scene_seed: int
    scene_id: str


@dataclass(frozen=True)
class CorpusPlan:
    plan_seed: int
    plan_version: str
    splits: tuple[str, ...]
    families: tuple[str, ...]
    totals: dict[str, dict[str, int]]
    assignments: tuple[SceneAssignment, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "plan_seed": self.plan_seed,
            "plan_version": self.plan_version,
            "splits": list(self.splits),
            "families": list(self.families),
            "totals": self.totals,
            "assignments": [asdict(a) for a in self.assignments],
        }

    @property
    def scene_count(self) -> int:
        return len(self.assignments)


PLAN_VERSION = "1"


def plan_corpus(
    plan_seed: int,
    totals: dict[str, dict[str, int]],
) -> CorpusPlan:
    """Return a deterministic :class:`CorpusPlan`.

    ``totals`` maps ``split -> {family -> count}``. Splits and families are
    validated against the registered set so a typo fails loudly instead of
    silently producing an empty plan.
    """

    known_families = set(registered_families())
    assignments: list[SceneAssignment] = []
    seen_seeds: set[int] = set()

    splits_present: list[str] = []
    families_present: set[str] = set()

    for split, family_counts in totals.items():
        if not isinstance(family_counts, dict):
            raise TypeError(f"totals[{split!r}] must be a dict of family->count")
        splits_present.append(split)
        for family, count in family_counts.items():
            if family not in known_families:
                raise ValueError(
                    f"unknown family '{family}' in totals[{split!r}]; "
                    f"registered families: {sorted(known_families)}"
                )
            if int(count) < 0:
                raise ValueError(f"negative count for totals[{split!r}][{family!r}]")
            families_present.add(family)
            for index in range(int(count)):
                seed = _scene_seed(plan_seed=plan_seed, split=split, family=family, index=index)
                if seed in seen_seeds:  # extremely unlikely with 64-bit hashes
                    seed = (seed + 1) & 0xFFFFFFFFFFFFFFFF
                seen_seeds.add(seed)
                assignments.append(
                    SceneAssignment(
                        split=split,
                        family=family,
                        scene_index=index,
                        scene_seed=seed,
                        scene_id=stable_scene_id(family, seed),
                    )
                )

    return CorpusPlan(
        plan_seed=int(plan_seed),
        plan_version=PLAN_VERSION,
        splits=tuple(splits_present),
        families=tuple(sorted(families_present)),
        totals={s: dict(f) for s, f in totals.items()},
        assignments=tuple(assignments),
    )


def smoke_corpus_plan(plan_seed: int = 0) -> CorpusPlan:
    """One scene per family across train/val/test_id, plus one hard test scene.

    Matches the Phase 8 smoke tier scale (~10 scenes) while exercising every
    registered family and every split. Useful as a CI/local sanity check.
    """

    totals: dict[str, dict[str, int]] = {
        "train": {family: 1 for family in registered_families()},
        "val": {family: 1 for family in ("open_obstacle_field", "medium_enclosed_maze")},
        "test_id": {family: 1 for family in ("medium_enclosed_maze", "large_enclosed_maze")},
        "test_hard": {"loop_alias_stress": 1},
    }
    return plan_corpus(plan_seed=plan_seed, totals=totals)


def standard_corpus_plan(
    *,
    plan_seed: int = 1,
    train_scenes: int = 200,
    val_scenes: int = 50,
    test_id_scenes: int = 50,
    test_hard_scenes: int = 50,
) -> CorpusPlan:
    """Spec-aligned plan with configurable per-split totals.

    Counts are distributed across families using :data:`TRAIN_SHARES` for
    train/val/test_id and :data:`HARD_TEST_SHARES` for the hard test split.
    Fractional allocations are rounded with leftovers given to the family
    holding the largest share so that the totals add up exactly.
    """

    totals: dict[str, dict[str, int]] = {
        "train": _allocate(train_scenes, TRAIN_SHARES),
        "val": _allocate(val_scenes, TRAIN_SHARES),
        "test_id": _allocate(test_id_scenes, TRAIN_SHARES),
        "test_hard": _allocate(test_hard_scenes, HARD_TEST_SHARES),
    }
    return plan_corpus(plan_seed=plan_seed, totals=totals)


def plan_sha256(plan: CorpusPlan) -> str:
    payload = json.dumps(plan.to_dict(), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _scene_seed(*, plan_seed: int, split: str, family: str, index: int) -> int:
    key = f"{plan_seed}:{split}:{family}:{index}".encode("utf-8")
    digest = hashlib.sha256(key).digest()
    return int.from_bytes(digest[:8], byteorder="big", signed=False)


def _allocate(total: int, shares: dict[str, float]) -> dict[str, int]:
    if total <= 0:
        return {family: 0 for family in shares}
    raw = {family: total * weight for family, weight in shares.items()}
    base = {family: int(value) for family, value in raw.items()}
    remainder = total - sum(base.values())
    if remainder:
        leftover = sorted(
            shares.items(),
            key=lambda item: (raw[item[0]] - base[item[0]], item[1]),
            reverse=True,
        )
        for family, _ in leftover[:remainder]:
            base[family] += 1
    return base
