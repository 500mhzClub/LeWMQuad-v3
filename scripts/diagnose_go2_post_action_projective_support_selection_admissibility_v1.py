#!/usr/bin/env python3
"""Census the corrected admissible-prefix target on checkpoint selection only."""
from __future__ import annotations

from collections import defaultdict
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Iterable, Mapping, MutableMapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
for package_root in (ROOT, ROOT / "lewm_worlds"):
    if str(package_root) not in sys.path:
        sys.path.insert(0, str(package_root))

from lewm.benchmarks import (  # noqa: E402
    go2_post_action_projective_support_labels_v1 as labels,
)


SCHEMA = "lewm_go2_post_action_projective_support_selection_admissibility_census_v1"
SELECTION_ROLE = "checkpoint_selection"
NON_HOLD_ACTIONS = tuple(action for action in labels.ACTION_ORDER if action != "hold")
V4_BINDING_BYTE_COUNT = 113_633
V4_BINDING_FILE_SHA256 = (
    "ec767a116cf9d0c231c6f7e5f18d6f6a9c6bb10206eea50e4063604fe707743a"
)
V4_BINDING_CONTENT_SHA256 = (
    "d0870f343e7a379a9627712ab2988a82e68ae35f0f5379f0c1ec753ce6bd1d86"
)


def _new_action_counts() -> dict[str, Any]:
    return {
        "state_count": 0,
        "immediate_feasible_count": 0,
        "blind_bridge_feasible_count": 0,
        "admissible_count": 0,
        "positive_remote_prefix_count": 0,
        "positive_admissible_prefix_count": 0,
        "remote_prefix_histogram_0_through_11": [0] * 12,
        "admissible_prefix_histogram_0_through_11": [0] * 12,
    }


def _new_scope() -> dict[str, Any]:
    return {
        "state_count": 0,
        "original_conjunct_counts": {
            "primary_subset_eligible": 0,
            "positive_best_remote_safe_prefix": 0,
            "at_least_two_distinct_nonhold_remote_safe_prefixes": 0,
            "informative_state": 0,
        },
        "proposed_conjunct_counts": {
            "positive_best_admissible_prefix": 0,
            "at_least_two_distinct_nonhold_admissible_prefixes": 0,
            "informative_state": 0,
        },
        "proposed_rejection_counts": {
            "zero_best_admissible_prefix": 0,
            "positive_but_no_action_difference": 0,
        },
        "informative_transition_counts": {
            "original_and_proposed": 0,
            "proposed_only": 0,
            "original_only": 0,
            "neither": 0,
        },
        "actions": {action: _new_action_counts() for action in NON_HOLD_ACTIONS},
    }


def _increment_scope(
    scope: MutableMapping[str, Any],
    *,
    by_action: Mapping[str, Mapping[str, Any]],
) -> None:
    remote = {
        action: int(row["remote_safe_prefix_length"])
        for action, row in by_action.items()
    }
    admissible = {
        action: bool(row["immediate_primitive"]["feasible"])
        and bool(row["blind_bridge"]["feasible"])
        for action, row in by_action.items()
    }
    composed = {
        action: remote[action] if admissible[action] else 0 for action in by_action
    }
    primary = all(admissible.values())
    remote_positive = max(remote.values()) > 0
    remote_varied = len(set(remote.values())) >= 2
    original_informative = primary and remote_positive and remote_varied
    composed_positive = max(composed.values()) > 0
    composed_varied = len(set(composed.values())) >= 2
    proposed_informative = composed_positive and composed_varied

    scope["state_count"] += 1
    original = scope["original_conjunct_counts"]
    for name, value in (
        ("primary_subset_eligible", primary),
        ("positive_best_remote_safe_prefix", remote_positive),
        ("at_least_two_distinct_nonhold_remote_safe_prefixes", remote_varied),
        ("informative_state", original_informative),
    ):
        original[name] += int(value)
    proposed = scope["proposed_conjunct_counts"]
    for name, value in (
        ("positive_best_admissible_prefix", composed_positive),
        ("at_least_two_distinct_nonhold_admissible_prefixes", composed_varied),
        ("informative_state", proposed_informative),
    ):
        proposed[name] += int(value)
    if not composed_positive:
        scope["proposed_rejection_counts"]["zero_best_admissible_prefix"] += 1
    elif not composed_varied:
        scope["proposed_rejection_counts"]["positive_but_no_action_difference"] += 1
    transition = (
        "original_and_proposed"
        if original_informative and proposed_informative
        else "proposed_only"
        if proposed_informative
        else "original_only"
        if original_informative
        else "neither"
    )
    scope["informative_transition_counts"][transition] += 1

    for action, row in by_action.items():
        counts = scope["actions"][action]
        immediate = bool(row["immediate_primitive"]["feasible"])
        blind = bool(row["blind_bridge"]["feasible"])
        counts["state_count"] += 1
        counts["immediate_feasible_count"] += int(immediate)
        counts["blind_bridge_feasible_count"] += int(blind)
        counts["admissible_count"] += int(admissible[action])
        counts["positive_remote_prefix_count"] += int(remote[action] > 0)
        counts["positive_admissible_prefix_count"] += int(composed[action] > 0)
        counts["remote_prefix_histogram_0_through_11"][remote[action]] += 1
        counts["admissible_prefix_histogram_0_through_11"][composed[action]] += 1


def aggregate_selection_rows_v1(
    rows: Sequence[Mapping[str, Any]], *, families: Iterable[str]
) -> dict[str, Any]:
    family_order = tuple(families)
    if not family_order or len(set(family_order)) != len(family_order):
        raise labels.LabelContractError("census family registry is empty or repeated")
    family_scopes = {family: _new_scope() for family in family_order}
    grouped: dict[int, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        if row.get("dataset_role") != SELECTION_ROLE:
            raise labels.LabelContractError("census row escaped checkpoint selection")
        grouped[int(row["role_state_index"])].append(row)

    aggregate = _new_scope()
    for state_index in sorted(grouped):
        state_rows = sorted(grouped[state_index], key=lambda row: int(row["action_index"]))
        if [row.get("action") for row in state_rows] != list(labels.ACTION_ORDER):
            raise labels.LabelContractError("census state action set/order changed")
        family = str(state_rows[0].get("family"))
        if family not in family_scopes or any(row.get("family") != family for row in state_rows):
            raise labels.LabelContractError("census state family escaped its registry")
        by_action = {
            str(row["action"]): row for row in state_rows if row["action"] != "hold"
        }
        primary = all(
            bool(row["immediate_primitive"]["feasible"])
            and bool(row["blind_bridge"]["feasible"])
            for row in by_action.values()
        )
        prefixes = [int(row["remote_safe_prefix_length"]) for row in by_action.values()]
        original_informative = primary and max(prefixes) > 0 and len(set(prefixes)) >= 2
        if any(
            row.get("primary_subset_eligible") is not primary
            or row.get("informative_state") is not original_informative
            for row in state_rows
        ):
            raise labels.LabelContractError("census rows disagree with frozen label logic")
        _increment_scope(aggregate, by_action=by_action)
        _increment_scope(family_scopes[family], by_action=by_action)

    aggregate["family_count"] = len(family_order)
    return {"aggregate": aggregate, "families": family_scopes}


def _path(record: Mapping[str, Any]) -> Path:
    candidate = Path(str(record["path"]))
    return (candidate if candidate.is_absolute() else ROOT / candidate).absolute()


def _assert_bound_size(binding: Mapping[str, Any], name: str, path: Path) -> None:
    record = binding["inputs"][name]
    if path.stat().st_size != int(record["byte_count"]):
        raise labels.LabelContractError(f"bound input byte count changed: {name}")


def run_census_v1() -> Mapping[str, Any]:
    ledger = labels.new_access_ledger_v1()
    binding_path = ROOT / labels.LABEL_EXECUTION_BINDING_RELATIVE_PATH
    raw_binding = binding_path.read_bytes()
    ledger["execution_binding_opens"] += 1
    if (
        len(raw_binding) != V4_BINDING_BYTE_COUNT
        or hashlib.sha256(raw_binding).hexdigest() != V4_BINDING_FILE_SHA256
    ):
        raise labels.LabelContractError("exact V4 execution binding bytes changed")
    binding = json.loads(raw_binding)
    if (
        not isinstance(binding, dict)
        or labels.canonical_json_bytes(binding) != raw_binding
        or binding.get("content_sha256") != V4_BINDING_CONTENT_SHA256
    ):
        raise labels.LabelContractError("exact V4 execution binding content changed")
    labels.validate_execution_binding_envelope_v1(binding)
    inputs = {name: _path(record) for name, record in binding["inputs"].items()}

    raw_indexes = labels.load_and_validate_raw_indexes(
        inputs["raw_manifest"],
        inputs["raw_pairs"],
        inputs["raw_endpoints"],
        access_ledger=ledger,
    )
    for name in ("raw_manifest", "raw_pairs", "raw_endpoints"):
        _assert_bound_size(binding, name, inputs[name])
    labels.validate_raw_audit_v1(inputs["raw_audit"], access_ledger=ledger)
    _assert_bound_size(binding, "raw_audit", inputs["raw_audit"])
    source_records = labels.validate_execution_binding_v1(binding, raw_indexes=raw_indexes)
    geometry = labels.load_geometry_inputs_v1(
        repository_root=ROOT,
        geometry_path=inputs["geometry_contract"],
        directional_policy_path=inputs["directional_policy"],
        primitive_registry_path=inputs["primitive_registry"],
        access_ledger=ledger,
    )
    for name in ("geometry_contract", "directional_policy", "primitive_registry"):
        _assert_bound_size(binding, name, inputs[name])

    selection_pairs = tuple(
        pair for pair in raw_indexes.pairs if pair["dataset_role"] == SELECTION_ROLE
    )
    role_index = {
        str(pair["content_sha256"]): index for index, pair in enumerate(selection_pairs)
    }
    selection_scenes = sorted(
        scene
        for scene, shard in raw_indexes.shard_by_scene.items()
        if shard["dataset_role"] == SELECTION_ROLE
    )
    if len(selection_scenes) != 8:
        raise labels.LabelContractError("checkpoint-selection scene count changed")
    rows: list[dict[str, Any]] = []
    for scene_id in selection_scenes:
        scene_manifest, states = labels.load_joined_scene_v1(
            raw_indexes=raw_indexes,
            scene_id=scene_id,
            source_records=source_records[scene_id],
            repository_root=ROOT,
            access_ledger=ledger,
        )
        for state in states:
            pair_sha256 = str(state.pair["content_sha256"])
            rows.extend(
                labels.label_state_v1(
                    pair=state.pair,
                    endpoint=state.endpoint,
                    source_pose_world=state.source_pose_world,
                    source_line_number=state.source_line_number,
                    scene_manifest=scene_manifest,
                    footprint=geometry.footprint,
                    commands_by_action=geometry.commands_by_action,
                    source_bindings=state.source_bindings,
                    role_state_index=role_index[pair_sha256],
                )
            )
    rows.sort(key=lambda row: (int(row["role_state_index"]), int(row["action_index"])))
    labels.validate_label_rows_v1(rows, role=SELECTION_ROLE)
    census = aggregate_selection_rows_v1(
        rows, families=labels.REGISTERED_SELECTION_FAMILIES
    )

    expected_ledger = labels.new_access_ledger_v1()
    expected_ledger.update(
        {
            "execution_binding_opens": 1,
            "raw_manifest_opens": 1,
            "raw_pairs_opens": 1,
            "raw_endpoints_opens": 1,
            "raw_audit_opens": 1,
            "geometry_contract_opens": 1,
            "geometry_contract_validation_calls": 1,
            "directional_policy_opens": 1,
            "primitive_registry_opens": 1,
            "scene_join_calls_started": 8,
            "render_summary_opens": 8,
            "source_frames_jsonl_opens": 8,
            "scene_manifest_opens": 8,
        }
    )
    if ledger != expected_ledger:
        raise labels.LabelContractError("selection census access ledger changed")
    aggregate = census["aggregate"]
    family_counts = {
        family: value["proposed_conjunct_counts"]["informative_state"]
        for family, value in census["families"].items()
    }
    passes = (
        aggregate["proposed_conjunct_counts"]["informative_state"] >= 128
        and all(count >= 8 for count in family_counts.values())
    )
    return labels.with_content_sha256(
        {
            "schema": SCHEMA,
            "status": "PASS_SELECTION_SCREEN" if passes else "STOP_SELECTION_SCREEN",
            "decision_rule": {
                "minimum_proposed_informative_states": 128,
                "minimum_proposed_informative_states_per_registered_family": 8,
            },
            "binding": {
                "path": labels.LABEL_EXECUTION_BINDING_RELATIVE_PATH,
                "byte_count": V4_BINDING_BYTE_COUNT,
                "file_sha256": V4_BINDING_FILE_SHA256,
                "content_sha256": V4_BINDING_CONTENT_SHA256,
            },
            "selection_state_count": len(selection_pairs),
            "selection_action_row_count": len(rows),
            "selection_scene_count": len(selection_scenes),
            "census": census,
            "access_ledger": ledger,
            "authority": {
                "rgb_opened": False,
                "model_or_checkpoint_opened": False,
                "gpu_or_training_used": False,
                "schedule_opened": False,
                "labels_or_v4_output_opened": False,
                "g2_heldout_or_sealed_opened": False,
                "filesystem_outputs_written": False,
            },
        }
    )


def main() -> int:
    if len(sys.argv) != 1:
        raise SystemExit("this one-shot diagnostic accepts no arguments")
    print(json.dumps(run_census_v1(), sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
