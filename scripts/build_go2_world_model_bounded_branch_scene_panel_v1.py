#!/usr/bin/env python3
"""Derive the blind bounded-pilot scene panel from the ordinary TRAIN universe.

The builder enumerates every direct campaign ``corpus.json`` below the ordinary
scene-corpus root, removes caller-independent frozen exclusion IDs, and applies
one fixed SHA-256 ranking.  It cannot accept hand-picked scenes or role labels.
Only metadata for the 32 selected scenes is opened beyond the corpus manifests.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from lewm.benchmarks import go2_world_model_counterfactual_pilot_v1 as pilot  # noqa: E402
from scripts import render_replay_v03 as reference_renderer  # noqa: E402


PANEL_SCHEMA = "lewm_go2_world_model_bounded_branch_scene_panel_v2"
SELECTION_SCHEMA = "lewm_go2_world_model_bounded_branch_scene_selection_v1"
SCENE_CORPUS_ROOT = REPO_ROOT / ".generated/scene_corpus"
SELECTION_SEED = 20260802
SCENES_PER_FAMILY = 4
SCENES_PER_ROLE_PER_FAMILY = 2
HISTORY_PANEL = (
    (3, 3),
    (5, 5),
    (6, 6),
    (7, 7),
    (8, 8),
    (0, 1),
    (2, 3),
    (4, 5),
)


class BoundedBranchScenePanelError(RuntimeError):
    """Raised before a biased or mutable scene panel can be emitted."""


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


def _corpus_manifests() -> list[Path]:
    root = SCENE_CORPUS_ROOT
    if root.is_symlink() or not root.is_dir():
        raise BoundedBranchScenePanelError("ordinary scene-corpus root is unavailable")
    manifests = []
    for campaign in sorted(root.iterdir(), key=lambda path: path.name):
        if _protected(campaign):
            continue
        if campaign.is_symlink() or not campaign.is_dir():
            continue
        candidate = campaign / "corpus.json"
        if candidate.is_symlink():
            raise BoundedBranchScenePanelError("corpus manifest is a symlink")
        if candidate.is_file():
            manifests.append(candidate)
    if not manifests:
        raise BoundedBranchScenePanelError("ordinary corpus inventory is empty")
    return manifests


def _load_inventory() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    candidates: dict[str, dict[str, Any]] = {}
    corpus_bindings = []
    for corpus_path in _corpus_manifests():
        binding = pilot.file_binding(corpus_path)
        value, actual = pilot.read_bound_json(
            corpus_path,
            expected_sha256=str(binding["file_sha256"]),
            expected_byte_count=int(binding["byte_count"]),
            label="ordinary scene corpus manifest",
        )
        if actual != binding or not isinstance(value, Mapping) or set(value) != {
            "plan",
            "plan_sha256",
            "scenes",
        }:
            raise BoundedBranchScenePanelError("ordinary corpus manifest changed")
        scenes = value["scenes"]
        if not isinstance(scenes, list):
            raise BoundedBranchScenePanelError("ordinary corpus scene inventory changed")
        corpus_bindings.append(binding)
        for row in scenes:
            if not isinstance(row, Mapping) or row.get("split") != "train":
                continue
            family = row.get("family")
            scene_id = row.get("scene_id")
            relative = row.get("relative_dir")
            manifest_sha = row.get("manifest_sha256")
            if (
                family not in pilot.FAMILIES
                or not isinstance(scene_id, str)
                or not scene_id
                or not isinstance(relative, str)
                or Path(relative).is_absolute()
                or ".." in Path(relative).parts
                or Path(relative).parts
                != ("train", str(family), scene_id)
                or not isinstance(manifest_sha, str)
                or len(manifest_sha) != 64
            ):
                raise BoundedBranchScenePanelError(
                    "ordinary TRAIN corpus row is malformed"
                )
            candidate = {
                "family": str(family),
                "scene_id": scene_id,
                "manifest_sha256": manifest_sha,
                "campaign_root": str(corpus_path.parent.resolve()),
                "relative_dir": relative,
                "inventory_rank": _rank(
                    "inventory-dedup",
                    family,
                    scene_id,
                    manifest_sha,
                    corpus_path.parent.name,
                ),
            }
            previous = candidates.get(scene_id)
            if previous is None or candidate["inventory_rank"] < previous["inventory_rank"]:
                candidates[scene_id] = candidate
    return corpus_bindings, list(candidates.values())


def _validated_selected_scene_root(
    *, campaign_root: str, relative_dir: str
) -> Path:
    """Reject ancestor symlinks before any selected-scene file is opened."""

    ordinary_root = SCENE_CORPUS_ROOT.resolve(strict=True)
    campaign = Path(campaign_root)
    relative = Path(relative_dir)
    if (
        not campaign.is_absolute()
        or campaign.is_symlink()
        or not campaign.is_dir()
        or campaign.resolve(strict=True) != campaign
        or campaign.parent != ordinary_root
        or relative.is_absolute()
        or ".." in relative.parts
    ):
        raise BoundedBranchScenePanelError(
            "selected scene campaign path is not an exact ordinary corpus directory"
        )
    cursor = campaign
    for part in relative.parts:
        cursor = cursor / part
        if cursor.is_symlink():
            raise BoundedBranchScenePanelError(
                "selected scene path contains a symlink"
            )
        if not cursor.is_dir():
            raise BoundedBranchScenePanelError(
                "selected scene path component is not a directory"
            )
    if cursor.resolve(strict=True) != cursor:
        raise BoundedBranchScenePanelError(
            "selected scene path is not canonical"
        )
    return cursor


def _selected_texture_asset_bindings(
    manifest_document: Mapping[str, Any],
) -> dict[str, dict[str, Any]]:
    selected = reference_renderer.select_scene_textures(
        visual_seed=int(manifest_document.get("visual_seed") or 0),
        scene_id=str(manifest_document.get("scene_id") or ""),
    )
    categories = ("floor", "wall", "obstacle")
    if not isinstance(selected, Mapping) or set(selected) != set(categories):
        raise BoundedBranchScenePanelError(
            "reference texture selection categories changed"
        )
    result = {}
    for category in categories:
        path_value = selected[category]
        if not isinstance(path_value, str) or not path_value:
            raise BoundedBranchScenePanelError(
                "reference textured_v03 asset selection is incomplete"
            )
        asset_path = Path(path_value)
        category_root = (REPO_ROOT / "assets/textures" / category).resolve()
        try:
            relative = asset_path.resolve(strict=True).relative_to(category_root)
        except (OSError, ValueError) as exc:
            raise BoundedBranchScenePanelError(
                "reference texture asset escaped its exact category"
            ) from exc
        if (
            asset_path.is_symlink()
            or len(relative.parts) != 1
            or relative.suffix.lower() not in {".jpg", ".jpeg", ".png"}
        ):
            raise BoundedBranchScenePanelError(
                "reference texture asset is not a category leaf image"
            )
        result[category] = pilot.file_binding(asset_path)
    return result


def derive_scene_panel_v1(*, excluded_scene_ids: set[str]) -> dict[str, Any]:
    if any(not isinstance(scene_id, str) or not scene_id for scene_id in excluded_scene_ids):
        raise BoundedBranchScenePanelError("scene exclusion set is malformed")
    corpus_bindings, inventory = _load_inventory()
    eligible_by_family: dict[str, list[dict[str, Any]]] = {
        family: [] for family in pilot.FAMILIES
    }
    for row in inventory:
        if row["scene_id"] in excluded_scene_ids:
            continue
        ranked = dict(row)
        ranked["selection_rank"] = _rank(
            "scene-selection",
            row["family"],
            row["scene_id"],
            row["manifest_sha256"],
        )
        eligible_by_family[row["family"]].append(ranked)
    selected: list[dict[str, Any]] = []
    eligible_counts = {}
    for family in pilot.FAMILIES:
        eligible = sorted(
            eligible_by_family[family],
            key=lambda row: (row["selection_rank"], row["scene_id"]),
        )
        eligible_counts[family] = len(eligible)
        if len(eligible) < SCENES_PER_FAMILY:
            raise BoundedBranchScenePanelError(
                f"family {family} has fewer than four model-disjoint scenes"
            )
        chosen = eligible[:SCENES_PER_FAMILY]
        chosen.sort(
            key=lambda row: (
                _rank("role-allocation", family, row["scene_id"]),
                row["scene_id"],
            )
        )
        for position, row in enumerate(chosen):
            role = (
                "train"
                if position < SCENES_PER_ROLE_PER_FAMILY
                else "eval"
            )
            scene_slot = position % SCENES_PER_ROLE_PER_FAMILY
            scene_root = _validated_selected_scene_root(
                campaign_root=str(row["campaign_root"]),
                relative_dir=str(row["relative_dir"]),
            )
            manifest_path = scene_root / "manifest.json"
            genesis_path = scene_root / "genesis_scene.json"
            if _protected(manifest_path) or _protected(genesis_path):
                raise BoundedBranchScenePanelError("selected scene names protected material")
            manifest_binding = pilot.file_binding(manifest_path)
            manifest_document, manifest_actual = pilot.read_bound_json(
                manifest_path,
                expected_sha256=str(manifest_binding["file_sha256"]),
                expected_byte_count=int(manifest_binding["byte_count"]),
                label=f"selected ordinary scene {row['scene_id']} manifest",
            )
            genesis_binding = pilot.file_binding(genesis_path)
            if (
                manifest_actual != manifest_binding
                or not isinstance(manifest_document, Mapping)
                or manifest_document.get("scene_id") != row["scene_id"]
                or manifest_document.get("family") != family
                or manifest_document.get("split") != "train"
                or manifest_document.get("manifest_sha256")
                != row["manifest_sha256"]
            ):
                raise BoundedBranchScenePanelError(
                    "selected scene manifest changed from its corpus inventory"
                )
            if _validated_selected_scene_root(
                campaign_root=str(row["campaign_root"]),
                relative_dir=str(row["relative_dir"]),
            ) != scene_root:
                raise BoundedBranchScenePanelError(
                    "selected scene path changed while it was bound"
                )
            selected_texture_assets = _selected_texture_asset_bindings(
                manifest_document
            )
            selected.append(
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
                    "selected_texture_asset_bindings": selected_texture_assets,
                    "states": [
                        {
                            "state_id": (
                                f"bounded-{role}-{row['scene_id']}-state-{index}"
                            ),
                            "history_action_ids": list(history),
                        }
                        for index, history in enumerate(HISTORY_PANEL)
                    ],
                }
            )
    selected.sort(
        key=lambda row: (
            ("train", "eval").index(row["role"]),
            pilot.FAMILIES.index(row["family"]),
            row["scene_slot"],
        )
    )
    return {
        "schema": PANEL_SCHEMA,
        "selection_contract": {
            "schema": SELECTION_SCHEMA,
            "seed": SELECTION_SEED,
            "universe": "all_direct_ordinary_scene_corpus_campaign_manifests",
            "eligible_split": "train",
            "deduplication": "lowest_sha256_inventory_rank_per_scene_id",
            "selection": "lowest_four_sha256_ranks_per_family",
            "role_allocation": "lowest_two_of_four_role_hashes_train_remainder_eval",
            "caller_scene_selection_allowed": False,
        },
        "corpus_manifest_bindings": corpus_bindings,
        "inventory_unique_train_scenes": len(inventory),
        "eligible_counts_by_family": eligible_counts,
        "excluded_scene_ids_sha256": hashlib.sha256(
            _canonical_bytes(sorted(excluded_scene_ids))
        ).hexdigest(),
        "scenes": selected,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--excluded-scene-ids", required=True, type=Path)
    parser.add_argument("--expected-exclusions-sha256", required=True)
    parser.add_argument("--expected-exclusions-byte-count", required=True, type=int)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    exclusions, _binding = pilot.read_bound_json(
        args.excluded_scene_ids,
        expected_sha256=args.expected_exclusions_sha256,
        expected_byte_count=args.expected_exclusions_byte_count,
        label="bounded scene exclusions",
    )
    if (
        not isinstance(exclusions, Mapping)
        or set(exclusions) != {"schema", "scene_ids"}
        or exclusions.get("schema")
        != "lewm_go2_world_model_bounded_branch_scene_exclusions_v1"
        or not isinstance(exclusions.get("scene_ids"), list)
    ):
        raise BoundedBranchScenePanelError("scene exclusion document changed")
    panel = derive_scene_panel_v1(excluded_scene_ids=set(exclusions["scene_ids"]))
    binding = pilot.write_json_exclusive(args.output, panel)
    print(json.dumps({"scene_panel": binding}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "BoundedBranchScenePanelError",
    "HISTORY_PANEL",
    "PANEL_SCHEMA",
    "SELECTION_SEED",
    "derive_scene_panel_v1",
]
