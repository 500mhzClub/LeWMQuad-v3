#!/usr/bin/env python3
"""Build the fresh-scene recurrent-dynamics replication panel and plan.

This builder changes the bounded counterfactual experiment's scene allocation,
not its physical or rendering contract.  It deterministically selects ordinary
TRAIN-corpus scenes after excluding both the frozen predecessor exclusion set
and every scene used by the predecessor panel.  The resulting plan is a copy
of the frozen bounded plan with only its attempt/output/state/count fields
replaced.

The module only opens ordinary metadata and texture assets.  It never opens
RGB, checkpoints, collected branches, or protected evaluation material.
"""
from __future__ import annotations

import argparse
from collections import Counter
import copy
import hashlib
import json
import math
from pathlib import Path
import re
import sys
from typing import Any, Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from lewm.benchmarks import go2_world_model_counterfactual_pilot_v1 as pilot  # noqa: E402
from scripts import (  # noqa: E402
    build_go2_world_model_bounded_branch_scene_panel_v1
    as predecessor_panel_builder,
)
from scripts.build_go2_world_model_counterfactual_calibration_plan_v1 import (  # noqa: E402
    _canonical_manifest_target,
)


PANEL_SCHEMA = "lewm_go2_scene_diversity_recurrent_replication_panel_v1"
SELECTION_SCHEMA = (
    "lewm_go2_scene_diversity_recurrent_replication_scene_selection_v1"
)
SELECTION_SEED = 20260804
ROLE_NAMES = ("train", "eval")
SCENES_PER_FAMILY = 8
SCENES_PER_ROLE_PER_FAMILY = 4
STATES_PER_SCENE = 4
HISTORY_PANEL = predecessor_panel_builder.HISTORY_PANEL
EXPECTED_SCENES = len(pilot.FAMILIES) * SCENES_PER_FAMILY
EXPECTED_STATES = EXPECTED_SCENES * STATES_PER_SCENE
EXPECTED_BRANCHES = EXPECTED_STATES * pilot.ACTION_COUNT
EXPECTED_ROLE_COUNTS = {"eval": 128, "train": 128}

DEFAULT_ATTEMPT_ID = "go2-scene-diversity-recurrent-replication-v1"
DEFAULT_ATTEMPT_ROOT = (
    REPO_ROOT
    / ".generated/dev/go2_scene_diversity_recurrent_replication_v1/attempt_v1"
)
DEFAULT_OUTPUT_ROOT = DEFAULT_ATTEMPT_ROOT / "collection"
DEFAULT_PANEL_OUTPUT = DEFAULT_ATTEMPT_ROOT / "scene_panel.json"
DEFAULT_PLAN_OUTPUT = DEFAULT_ATTEMPT_ROOT / "exact_plan.json"

FROZEN_BASE_EXCLUSIONS = (
    REPO_ROOT
    / ".generated/dev/lewm-go2-wm-bounded-branch-metadata-v1/scene_exclusions.json"
)
FROZEN_BASE_EXCLUSIONS_SHA256 = (
    "fdb6eb7f0fac3768cd58c31321ae0c2b33143547896c9e2f2ab708d05127b30c"
)
FROZEN_BASE_EXCLUSIONS_BYTE_COUNT = 46824
FROZEN_PREDECESSOR_PANEL = (
    REPO_ROOT
    / ".generated/dev/lewm-go2-wm-bounded-branch-metadata-v1/scene_panel.json"
)
FROZEN_PREDECESSOR_PANEL_SHA256 = (
    "a5dad8e906ad594b7c3f7052ba2a2459e325fff2dfeeef11aeb9f333e7781994"
)
FROZEN_PREDECESSOR_PANEL_BYTE_COUNT = 111893
FROZEN_BASE_PLAN = (
    REPO_ROOT
    / (
        "docs/lewm_go2_world_model_bounded_branch_integrity_replacement_v1_"
        "exact_plan_2026-08-02.json"
    )
)
FROZEN_BASE_PLAN_SHA256 = (
    "8fe34054bb9ae709b6a8ecfea5fdae55c742d1b2e22af3c289d27a77f11c66ef"
)
FROZEN_BASE_PLAN_BYTE_COUNT = 343973

_SHA256 = re.compile(r"^[0-9a-f]{64}$")


class SceneDiversityPlanError(RuntimeError):
    """Raised before a biased panel or mutated physical contract is emitted."""


def _protected(path: Path) -> bool:
    return any(
        part.lower() == "sealed_test.json"
        or part.lower() == "sealed"
        or part.lower().startswith("sealed_")
        or part.lower() in {"heldout", "held_out", "held-out"}
        or part.lower().startswith("heldout_")
        or part.lower().startswith("held_out_")
        or part.lower().startswith("held-out-")
        for part in Path(path).parts
    )


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")


def _rank(namespace: str, *values: object) -> str:
    return hashlib.sha256(
        _canonical_bytes([PANEL_SCHEMA, SELECTION_SEED, namespace, *values])
    ).hexdigest()


def _ids_sha256(values: set[str]) -> str:
    return hashlib.sha256(_canonical_bytes(sorted(values))).hexdigest()


def _selection_contract() -> dict[str, Any]:
    return {
        "schema": SELECTION_SCHEMA,
        "seed": SELECTION_SEED,
        "universe": "all_direct_ordinary_scene_corpus_campaign_manifests",
        "eligible_split": "train",
        "deduplication": "lowest_sha256_inventory_rank_per_scene_id",
        "selection": "lowest_eight_fresh_sha256_ranks_per_family",
        "role_allocation": (
            "lowest_four_of_eight_role_hashes_train_remainder_eval"
        ),
        "states_per_scene": STATES_PER_SCENE,
        "history_allocation": (
            "even_history_panel_indices_in_scene_slots_0_2_and_odd_indices_1_3"
        ),
        "caller_scene_selection_allowed": False,
    }


def _validate_base_exclusions(value: object) -> set[str]:
    if (
        not isinstance(value, Mapping)
        or set(value) != {"schema", "scene_ids"}
        or value.get("schema")
        != "lewm_go2_world_model_bounded_branch_scene_exclusions_v1"
        or not isinstance(value.get("scene_ids"), list)
    ):
        raise SceneDiversityPlanError("frozen base scene exclusions changed")
    scene_ids = value["scene_ids"]
    if (
        any(not isinstance(scene_id, str) or not scene_id for scene_id in scene_ids)
        or len(scene_ids) != len(set(scene_ids))
    ):
        raise SceneDiversityPlanError("frozen base scene exclusions are malformed")
    return set(scene_ids)


def _validate_predecessor_scene_ids(value: object) -> set[str]:
    if (
        not isinstance(value, Mapping)
        or value.get("schema") != predecessor_panel_builder.PANEL_SCHEMA
        or not isinstance(value.get("scenes"), list)
    ):
        raise SceneDiversityPlanError("frozen predecessor scene panel changed")
    scenes = value["scenes"]
    if len(scenes) != 32:
        raise SceneDiversityPlanError("predecessor panel must contain 32 scenes")
    scene_ids: set[str] = set()
    balance: Counter[tuple[str, str]] = Counter()
    for row in scenes:
        if not isinstance(row, Mapping):
            raise SceneDiversityPlanError("predecessor panel row is malformed")
        scene_id = row.get("scene_id")
        role = row.get("role")
        family = row.get("family")
        if (
            not isinstance(scene_id, str)
            or not scene_id
            or scene_id in scene_ids
            or role not in ROLE_NAMES
            or family not in pilot.FAMILIES
        ):
            raise SceneDiversityPlanError("predecessor panel identity changed")
        scene_ids.add(scene_id)
        balance[(str(role), str(family))] += 1
    if any(
        balance[(role, family)] != 2
        for role in ROLE_NAMES
        for family in pilot.FAMILIES
    ):
        raise SceneDiversityPlanError("predecessor panel family/role balance changed")
    return scene_ids


def _history_indices_for_scene_slot(scene_slot: int) -> tuple[int, ...]:
    if type(scene_slot) is not int or not 0 <= scene_slot < 4:
        raise SceneDiversityPlanError("scene slot is outside the four-scene role grid")
    return tuple(range(scene_slot % 2, len(HISTORY_PANEL), 2))


def derive_scene_panel_v1(
    *,
    base_exclusions: Mapping[str, Any],
    base_exclusions_binding: Mapping[str, Any],
    predecessor_panel: Mapping[str, Any],
    predecessor_panel_binding: Mapping[str, Any],
) -> dict[str, Any]:
    """Derive and bind the fresh 64-scene panel without writing artifacts."""

    try:
        bound_exclusions = pilot.require_binding(
            base_exclusions_binding, label="frozen base scene exclusions"
        )
        bound_predecessor = pilot.require_binding(
            predecessor_panel_binding, label="frozen predecessor scene panel"
        )
    except pilot.PilotContractError as exc:
        raise SceneDiversityPlanError(str(exc)) from exc
    base_ids = _validate_base_exclusions(base_exclusions)
    predecessor_ids = _validate_predecessor_scene_ids(predecessor_panel)
    excluded_ids = base_ids | predecessor_ids
    try:
        corpus_bindings, inventory = predecessor_panel_builder._load_inventory()
    except (
        pilot.PilotContractError,
        predecessor_panel_builder.BoundedBranchScenePanelError,
    ) as exc:
        raise SceneDiversityPlanError(str(exc)) from exc

    eligible_by_family: dict[str, list[dict[str, Any]]] = {
        family: [] for family in pilot.FAMILIES
    }
    for source_row in inventory:
        if not isinstance(source_row, Mapping):
            raise SceneDiversityPlanError("ordinary scene inventory row is malformed")
        family = source_row.get("family")
        scene_id = source_row.get("scene_id")
        manifest_sha = source_row.get("manifest_sha256")
        if (
            family not in pilot.FAMILIES
            or not isinstance(scene_id, str)
            or not scene_id
            or not isinstance(manifest_sha, str)
            or _SHA256.fullmatch(manifest_sha) is None
        ):
            raise SceneDiversityPlanError("ordinary scene inventory identity changed")
        if scene_id in excluded_ids:
            continue
        ranked = dict(source_row)
        ranked["selection_rank"] = _rank(
            "scene-selection", family, scene_id, manifest_sha
        )
        eligible_by_family[str(family)].append(ranked)

    selected_rows: list[dict[str, Any]] = []
    eligible_counts: dict[str, int] = {}
    for family in pilot.FAMILIES:
        eligible = sorted(
            eligible_by_family[family],
            key=lambda row: (str(row["selection_rank"]), str(row["scene_id"])),
        )
        eligible_counts[family] = len(eligible)
        if len(eligible) < SCENES_PER_FAMILY:
            raise SceneDiversityPlanError(
                f"family {family} has fewer than eight fresh ordinary TRAIN scenes"
            )
        chosen = eligible[:SCENES_PER_FAMILY]
        chosen.sort(
            key=lambda row: (
                _rank("role-allocation", family, row["scene_id"]),
                str(row["scene_id"]),
            )
        )
        for position, row in enumerate(chosen):
            role = "train" if position < 4 else "eval"
            scene_slot = position % SCENES_PER_ROLE_PER_FAMILY
            try:
                scene_root = predecessor_panel_builder._validated_selected_scene_root(
                    campaign_root=str(row["campaign_root"]),
                    relative_dir=str(row["relative_dir"]),
                )
                manifest_path = scene_root / "manifest.json"
                genesis_path = scene_root / "genesis_scene.json"
                if _protected(manifest_path) or _protected(genesis_path):
                    raise SceneDiversityPlanError(
                        "selected scene names custody-protected material"
                    )
                manifest_binding = pilot.file_binding(manifest_path)
                manifest, actual = pilot.read_bound_json(
                    manifest_path,
                    expected_sha256=str(manifest_binding["file_sha256"]),
                    expected_byte_count=int(manifest_binding["byte_count"]),
                    label=f"fresh ordinary scene {row['scene_id']} manifest",
                )
                genesis_binding = pilot.file_binding(genesis_path)
            except (
                OSError,
                pilot.PilotContractError,
                predecessor_panel_builder.BoundedBranchScenePanelError,
            ) as exc:
                raise SceneDiversityPlanError(str(exc)) from exc
            if (
                actual != manifest_binding
                or not isinstance(manifest, Mapping)
                or manifest.get("scene_id") != row["scene_id"]
                or manifest.get("family") != family
                or manifest.get("split") != "train"
                or manifest.get("manifest_sha256") != row["manifest_sha256"]
            ):
                raise SceneDiversityPlanError(
                    "selected scene manifest changed from the corpus inventory"
                )
            try:
                target_xy = _canonical_manifest_target(
                    manifest_binding,
                    scene_id=str(row["scene_id"]),
                    family=family,
                )
                texture_bindings = (
                    predecessor_panel_builder._selected_texture_asset_bindings(
                        manifest
                    )
                )
            except (pilot.PilotContractError, RuntimeError) as exc:
                raise SceneDiversityPlanError(str(exc)) from exc
            history_indices = _history_indices_for_scene_slot(scene_slot)
            states = []
            for state_index, history_index in enumerate(history_indices):
                states.append(
                    {
                        "state_id": (
                            f"scene-diversity-{role}-{row['scene_id']}-state-{state_index}"
                        ),
                        "history_panel_index": history_index,
                        "history_action_ids": list(HISTORY_PANEL[history_index]),
                        "target_xy_m": list(target_xy),
                    }
                )
            selected_rows.append(
                {
                    "role": role,
                    "family": family,
                    "scene_slot": scene_slot,
                    "scene_id": row["scene_id"],
                    "inventory_manifest_sha256": row["manifest_sha256"],
                    "selection_rank": row["selection_rank"],
                    "role_allocation_rank": _rank(
                        "role-allocation", family, row["scene_id"]
                    ),
                    "scene_manifest_binding": manifest_binding,
                    "scene_genesis_binding": genesis_binding,
                    "selected_texture_asset_bindings": texture_bindings,
                    "states": states,
                }
            )

    selected_rows.sort(
        key=lambda row: (
            ROLE_NAMES.index(str(row["role"])),
            pilot.FAMILIES.index(str(row["family"])),
            int(row["scene_slot"]),
        )
    )
    return {
        "schema": PANEL_SCHEMA,
        "selection_contract": _selection_contract(),
        "source_bindings": {
            "base_scene_exclusions": bound_exclusions,
            "predecessor_scene_panel": bound_predecessor,
        },
        "corpus_manifest_bindings": copy.deepcopy(corpus_bindings),
        "inventory_unique_train_scenes": len(inventory),
        "eligible_counts_by_family": eligible_counts,
        "base_excluded_scene_ids_count": len(base_ids),
        "base_excluded_scene_ids_sha256": _ids_sha256(base_ids),
        "predecessor_scene_ids_count": len(predecessor_ids),
        "predecessor_scene_ids_sha256": _ids_sha256(predecessor_ids),
        "excluded_scene_ids_count": len(excluded_ids),
        "excluded_scene_ids_sha256": _ids_sha256(excluded_ids),
        "scenes": selected_rows,
    }


def _validate_panel_for_plan(value: object) -> list[dict[str, Any]]:
    required = {
        "schema",
        "selection_contract",
        "source_bindings",
        "corpus_manifest_bindings",
        "inventory_unique_train_scenes",
        "eligible_counts_by_family",
        "base_excluded_scene_ids_count",
        "base_excluded_scene_ids_sha256",
        "predecessor_scene_ids_count",
        "predecessor_scene_ids_sha256",
        "excluded_scene_ids_count",
        "excluded_scene_ids_sha256",
        "scenes",
    }
    if (
        not isinstance(value, Mapping)
        or set(value) != required
        or value.get("schema") != PANEL_SCHEMA
        or value.get("selection_contract") != _selection_contract()
        or not isinstance(value.get("scenes"), list)
        or len(value["scenes"]) != EXPECTED_SCENES
    ):
        raise SceneDiversityPlanError("scene-diversity panel contract changed")
    expected_order = [
        (role, family, slot)
        for role in ROLE_NAMES
        for family in pilot.FAMILIES
        for slot in range(SCENES_PER_ROLE_PER_FAMILY)
    ]
    normalized: list[dict[str, Any]] = []
    seen_scenes: set[str] = set()
    seen_states: set[str] = set()
    history_counts: Counter[tuple[str, str, tuple[int, int]]] = Counter()
    for expected, scene in zip(expected_order, value["scenes"], strict=True):
        if not isinstance(scene, Mapping) or set(scene) != {
            "role",
            "family",
            "scene_slot",
            "scene_id",
            "inventory_manifest_sha256",
            "selection_rank",
            "role_allocation_rank",
            "scene_manifest_binding",
            "scene_genesis_binding",
            "selected_texture_asset_bindings",
            "states",
        }:
            raise SceneDiversityPlanError("scene-diversity panel row changed")
        role, family, scene_slot = expected
        scene_id = scene.get("scene_id")
        if (
            scene.get("role") != role
            or scene.get("family") != family
            or scene.get("scene_slot") != scene_slot
            or not isinstance(scene_id, str)
            or not scene_id
            or scene_id in seen_scenes
            or _SHA256.fullmatch(str(scene.get("inventory_manifest_sha256"))) is None
            or _SHA256.fullmatch(str(scene.get("selection_rank"))) is None
            or _SHA256.fullmatch(str(scene.get("role_allocation_rank"))) is None
        ):
            raise SceneDiversityPlanError("scene-diversity panel order/identity changed")
        try:
            manifest_binding = pilot.require_binding(
                scene["scene_manifest_binding"], label=f"scene {scene_id} manifest"
            )
            genesis_binding = pilot.require_binding(
                scene["scene_genesis_binding"], label=f"scene {scene_id} Genesis pack"
            )
        except pilot.PilotContractError as exc:
            raise SceneDiversityPlanError(str(exc)) from exc
        if (
            Path(manifest_binding["path"]).name != "manifest.json"
            or Path(genesis_binding["path"]).name != "genesis_scene.json"
            or Path(manifest_binding["path"]).parent
            != Path(genesis_binding["path"]).parent
            or _protected(Path(manifest_binding["path"]))
            or _protected(Path(genesis_binding["path"]))
        ):
            raise SceneDiversityPlanError("scene manifest/Genesis binding pair changed")
        try:
            canonical_target = _canonical_manifest_target(
                manifest_binding, scene_id=scene_id, family=family
            )
        except (pilot.PilotContractError, RuntimeError) as exc:
            raise SceneDiversityPlanError(str(exc)) from exc
        textures = scene.get("selected_texture_asset_bindings")
        if not isinstance(textures, Mapping) or set(textures) != {
            "floor",
            "wall",
            "obstacle",
        }:
            raise SceneDiversityPlanError("selected texture binding set changed")
        for category in ("floor", "wall", "obstacle"):
            try:
                pilot.require_binding(
                    textures[category],
                    label=f"scene {scene_id} selected {category} texture",
                )
            except pilot.PilotContractError as exc:
                raise SceneDiversityPlanError(str(exc)) from exc
        states = scene.get("states")
        expected_history_indices = _history_indices_for_scene_slot(scene_slot)
        if not isinstance(states, list) or len(states) != STATES_PER_SCENE:
            raise SceneDiversityPlanError("each scene must contain four states")
        normalized_states = []
        for state_index, (state, history_index) in enumerate(
            zip(states, expected_history_indices, strict=True)
        ):
            if not isinstance(state, Mapping) or set(state) != {
                "state_id",
                "history_panel_index",
                "history_action_ids",
                "target_xy_m",
            }:
                raise SceneDiversityPlanError("scene-diversity state fields changed")
            state_id = state.get("state_id")
            history = list(HISTORY_PANEL[history_index])
            target = state.get("target_xy_m")
            if (
                not isinstance(state_id, str)
                or not state_id
                or state_id in seen_states
                or state.get("history_panel_index") != history_index
                or state.get("history_action_ids") != history
                or not isinstance(target, list)
                or len(target) != 2
                or any(
                    isinstance(coordinate, bool)
                    or not isinstance(coordinate, (int, float))
                    or not math.isfinite(float(coordinate))
                    for coordinate in target
                )
                or [float(coordinate) for coordinate in target] != canonical_target
            ):
                raise SceneDiversityPlanError(
                    "scene-diversity state identity/history/target changed"
                )
            seen_states.add(state_id)
            history_counts[(role, family, tuple(history))] += 1
            normalized_states.append(
                {
                    "state_id": state_id,
                    "history_action_ids": history,
                    "target_xy_m": canonical_target,
                }
            )
        seen_scenes.add(scene_id)
        normalized.append(
            {
                "role": role,
                "family": family,
                "scene_id": scene_id,
                "scene_manifest_binding": manifest_binding,
                "scene_genesis_binding": genesis_binding,
                "states": normalized_states,
            }
        )
    if any(
        history_counts[(role, family, tuple(history))] != 2
        for role in ROLE_NAMES
        for family in pilot.FAMILIES
        for history in HISTORY_PANEL
    ):
        raise SceneDiversityPlanError(
            "each history pair must occur exactly twice per role/family"
        )
    return normalized


def build_plan_v1(
    *,
    base_plan: Mapping[str, Any],
    attempt_id: str,
    output_root: Path,
    scene_panel: Mapping[str, Any],
) -> dict[str, Any]:
    """Copy the frozen physical plan and replace only replication identity/data."""

    try:
        normalized_base = pilot.validate_plan(base_plan)
    except pilot.PilotContractError as exc:
        raise SceneDiversityPlanError(str(exc)) from exc
    if (
        normalized_base != dict(base_plan)
        or normalized_base.get("purpose") != "bounded_wm_a_pilot"
        or normalized_base.get("states_per_scene") != 8
        or normalized_base.get("expected_counts", {}).get("scenes") != 32
        or normalized_base.get("expected_counts", {}).get("states") != 256
        or normalized_base.get("expected_counts", {}).get("roles")
        != EXPECTED_ROLE_COUNTS
    ):
        raise SceneDiversityPlanError("frozen bounded base plan contract changed")
    selected_root = Path(output_root)
    development_root = (REPO_ROOT / ".generated/dev").resolve()
    if (
        not selected_root.is_absolute()
        or not selected_root.resolve(strict=False).is_relative_to(development_root)
        or selected_root.exists()
        or selected_root.is_symlink()
        or _protected(selected_root)
    ):
        raise SceneDiversityPlanError(
            "output_root must be a fresh ordinary path under .generated/dev"
        )
    panel = _validate_panel_for_plan(scene_panel)
    states: list[dict[str, Any]] = []
    for scene in panel:
        for state_index, state in enumerate(scene["states"]):
            states.append(
                {
                    "state_id": state["state_id"],
                    "role": scene["role"],
                    "family": scene["family"],
                    "scene_id": scene["scene_id"],
                    "scene_manifest_binding": scene["scene_manifest_binding"],
                    "scene_genesis_binding": scene["scene_genesis_binding"],
                    "scene_generation": None,
                    "group_index": len(states),
                    "state_index_in_scene": state_index,
                    "history_action_ids": state["history_action_ids"],
                    "candidate_action_ids": list(range(pilot.ACTION_COUNT)),
                    "sentinel_duplicate_action_id": None,
                    "target_xy_m": state["target_xy_m"],
                }
            )
    plan = copy.deepcopy(normalized_base)
    plan["attempt_id"] = attempt_id
    plan["output_root"] = str(selected_root.resolve(strict=False))
    plan["states_per_scene"] = STATES_PER_SCENE
    plan["states"] = states
    plan["expected_counts"] = pilot.expected_counts_from_states(states)
    try:
        validated = pilot.validate_plan(plan)
    except pilot.PilotContractError as exc:
        raise SceneDiversityPlanError(str(exc)) from exc
    if validated != plan:
        raise SceneDiversityPlanError("pilot plan normalization changed the plan")
    counts = validated["expected_counts"]
    if (
        counts.get("scenes") != EXPECTED_SCENES
        or counts.get("states") != EXPECTED_STATES
        or counts.get("roles") != EXPECTED_ROLE_COUNTS
        or counts.get("candidate_branches") != EXPECTED_BRANCHES
        or counts.get("sentinel_branches") != 0
    ):
        raise SceneDiversityPlanError("replication plan count contract changed")
    mutable_fields = {
        "attempt_id",
        "output_root",
        "states_per_scene",
        "states",
        "expected_counts",
    }
    if any(
        _canonical_bytes(validated[field]) != _canonical_bytes(normalized_base[field])
        for field in set(normalized_base) - mutable_fields
    ):
        raise SceneDiversityPlanError("frozen runtime/render/action contract changed")
    return validated


def _load_frozen_json(
    path: Path, *, sha256: str, byte_count: int, label: str
) -> tuple[dict[str, Any], dict[str, Any]]:
    if _protected(path):
        raise SceneDiversityPlanError(f"{label} path is custody-protected")
    try:
        return pilot.read_bound_json(
            path,
            expected_sha256=sha256,
            expected_byte_count=byte_count,
            label=label,
        )
    except pilot.PilotContractError as exc:
        raise SceneDiversityPlanError(str(exc)) from exc


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--attempt-id", default=DEFAULT_ATTEMPT_ID)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--panel-output", type=Path, default=DEFAULT_PANEL_OUTPUT)
    parser.add_argument("--plan-output", type=Path, default=DEFAULT_PLAN_OUTPUT)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.panel_output == args.plan_output:
        raise SceneDiversityPlanError("panel and plan outputs must be different")
    if args.panel_output.exists() or args.plan_output.exists():
        raise SceneDiversityPlanError("panel and plan outputs must both be fresh")
    exclusions, exclusions_binding = _load_frozen_json(
        FROZEN_BASE_EXCLUSIONS,
        sha256=FROZEN_BASE_EXCLUSIONS_SHA256,
        byte_count=FROZEN_BASE_EXCLUSIONS_BYTE_COUNT,
        label="frozen base scene exclusions",
    )
    predecessor_panel, predecessor_panel_binding = _load_frozen_json(
        FROZEN_PREDECESSOR_PANEL,
        sha256=FROZEN_PREDECESSOR_PANEL_SHA256,
        byte_count=FROZEN_PREDECESSOR_PANEL_BYTE_COUNT,
        label="frozen predecessor scene panel",
    )
    base_plan, base_plan_binding = _load_frozen_json(
        FROZEN_BASE_PLAN,
        sha256=FROZEN_BASE_PLAN_SHA256,
        byte_count=FROZEN_BASE_PLAN_BYTE_COUNT,
        label="frozen bounded base plan",
    )
    panel = derive_scene_panel_v1(
        base_exclusions=exclusions,
        base_exclusions_binding=exclusions_binding,
        predecessor_panel=predecessor_panel,
        predecessor_panel_binding=predecessor_panel_binding,
    )
    plan = build_plan_v1(
        base_plan=base_plan,
        attempt_id=args.attempt_id,
        output_root=args.output_root,
        scene_panel=panel,
    )
    panel_binding = pilot.write_json_exclusive(args.panel_output, panel)
    plan_binding = pilot.write_json_exclusive(args.plan_output, plan)
    print(
        json.dumps(
            {
                "scene_panel": panel_binding,
                "plan": plan_binding,
                "frozen_base_plan": base_plan_binding,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "DEFAULT_ATTEMPT_ID",
    "DEFAULT_OUTPUT_ROOT",
    "EXPECTED_BRANCHES",
    "EXPECTED_SCENES",
    "EXPECTED_STATES",
    "HISTORY_PANEL",
    "PANEL_SCHEMA",
    "SELECTION_SEED",
    "STATES_PER_SCENE",
    "SceneDiversityPlanError",
    "build_plan_v1",
    "derive_scene_panel_v1",
]
