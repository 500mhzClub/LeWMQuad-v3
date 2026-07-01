#!/usr/bin/env python3
"""Diagnose Phase 3A learned marker-memory localization."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from lewm.benchmarks.phase3a_explore_claim import explore_claim_phase  # noqa: E402
from lewm.benchmarks.phase3a_positive_control import read_jsonl  # noqa: E402
from lewm.benchmarks.phase3a_training import (  # noqa: E402
    Phase3AMaterializedDataset,
    source_grouped_batches,
    source_key,
)
from scripts.report_jepa_phase3a_explore_claim import load_model  # noqa: E402


@torch.no_grad()
def diagnose(
    *,
    checkpoint: Path,
    validation_data: Path,
    output: Path,
    source_states_per_batch: int,
    device: torch.device,
) -> dict:
    rows = read_jsonl(validation_data)
    model, _ = load_model(checkpoint, device=device)
    cache = Phase3AMaterializedDataset(rows)
    grouped: dict[tuple[str, int], list[int]] = {}
    for index, row in enumerate(rows):
        grouped.setdefault(source_key(row), []).append(index)

    totals = {
        "rows": 0,
        "valid": 0,
        "top1": 0,
        "mean_mass_num": 0.0,
        "mean_target_prob_num": 0.0,
    }
    phases: dict[str, dict] = {}
    examples = []
    for indices in source_grouped_batches(
        rows,
        source_states_per_batch=source_states_per_batch,
        shuffle=False,
    ):
        batch = cache.materialize_batch(indices).to(device)
        output_batch = model(
            vision=batch.vision,
            history_vision=batch.history_vision,
            history_actions=batch.history_actions,
            actions=batch.actions,
            utility_targets=batch.utility_targets,
            consequence_targets=batch.consequence_targets,
            structured_marker_memory_valid_mask=batch.marker_memory_start_valid_mask,
            structured_marker_memory_start_delta_targets=(
                batch.marker_memory_start_delta_targets
            ),
            spatial_frontier_history_observation_targets=(
                batch.spatial_frontier_history_observation_targets
            ),
            spatial_frontier_vision_observation_targets=(
                batch.spatial_frontier_vision_observation_targets
            ),
            utility_group_ids=batch.utility_group_ids,
            utility_mask=batch.utility_mask,
            wrong_actions=batch.wrong_actions,
            wrong_mask=batch.wrong_mask,
            non_hold_mask=batch.non_hold_mask,
            return_latents=True,
        )
        belief = output_batch["spatial_frontier_marker_belief"]
        predicted_cells = belief.argmax(dim=-1)
        valid, targets = model.spatial_marker_memory_target_indices(
            batch.marker_memory_start_valid_mask,
            batch.marker_memory_start_delta_targets,
        )
        marker_mass = output_batch["spatial_frontier_marker_mass"]
        for local_index, row_index in enumerate(indices):
            row = rows[row_index]
            phase = explore_claim_phase([rows[item] for item in grouped[source_key(row)]])
            phase_totals = phases.setdefault(
                phase,
                {
                    "rows": 0,
                    "valid": 0,
                    "top1": 0,
                    "mean_mass_num": 0.0,
                    "mean_target_prob_num": 0.0,
                },
            )
            mass = float(marker_mass[local_index].detach().cpu())
            for item in (totals, phase_totals):
                item["rows"] += 1
                item["mean_mass_num"] += mass
            if bool(valid[local_index].detach().cpu()):
                target = int(targets[local_index].detach().cpu())
                predicted = int(predicted_cells[local_index].detach().cpu())
                target_probability = float(
                    belief[local_index, target].detach().cpu()
                )
                ok = int(predicted == target)
                for item in (totals, phase_totals):
                    item["valid"] += 1
                    item["top1"] += ok
                    item["mean_target_prob_num"] += target_probability
                if not ok and len(examples) < 12:
                    examples.append(
                        {
                            "source_key": f"{source_key(row)[0]}:{source_key(row)[1]}",
                            "phase": phase,
                            "target_cell": target,
                            "predicted_cell": predicted,
                            "target_probability": target_probability,
                            "marker_mass": mass,
                        }
                    )

    def finish(item: dict) -> dict:
        rows_count = max(int(item["rows"]), 1)
        valid_count = max(int(item["valid"]), 1)
        return {
            "rows": int(item["rows"]),
            "valid_marker_rows": int(item["valid"]),
            "top1_accuracy": float(item["top1"]) / float(valid_count),
            "mean_marker_mass": float(item["mean_mass_num"]) / float(rows_count),
            "mean_target_probability": (
                float(item["mean_target_prob_num"]) / float(valid_count)
            ),
        }

    report = {
        "schema": "jepa_phase3a_marker_memory_diagnostic_v0",
        "checkpoint": str(checkpoint.resolve()),
        "validation_data": str(validation_data.resolve()),
        "device": str(device),
        "overall": finish(totals),
        "phases": {phase: finish(phases[phase]) for phase in sorted(phases)},
        "mislocalized_examples": examples,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--validation-data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-states-per-batch", type=int, default=4)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    args = parser.parse_args()
    report = diagnose(
        checkpoint=args.checkpoint,
        validation_data=args.validation_data,
        output=args.output,
        source_states_per_batch=args.source_states_per_batch,
        device=torch.device(args.device),
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
