#!/usr/bin/env python3
"""Train a Phase 3A policy head over egocentric latent memory."""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from lewm.benchmarks.phase3a_positive_control import ACTION_NAMES, read_jsonl  # noqa: E402
from lewm.benchmarks.phase3a_positive_control import step_state  # noqa: E402
from lewm.models.phase3a_latent_map import (  # noqa: E402
    Phase3AEgocentricMemoryPolicy,
    Phase3AEgocentricMemoryUpdate,
)
from scripts.export_jepa_phase3a_closed_loop_demo_mp4 import (  # noqa: E402
    _action_index_tensor,
    _center_local_evidence,
    _egocentric_memory_tensor_to_dict,
    _goal_scene_from_row,
    _latent_local_evidence,
    _select_egocentric_frontier_action,
    _select_odom_frontier_action,
    _select_odom_frontier_lookahead_action,
    _state_from_dict,
    _update_odom_frontier_memory,
)
from scripts.train_jepa_phase3a_latent_memory import (  # noqa: E402
    _build_examples,
    _infer_scene_seed,
    _load_latent_map_head,
    _memory_from_tensor,
    _random_memory_tensor,
)
from scripts.report_jepa_phase3a_explore_claim import load_model  # noqa: E402


def _teacher_scores_from_memory(memory: dict) -> torch.Tensor:
    teacher_action = _select_egocentric_frontier_action(
        {
            "free": set(memory["free"]),
            "blocked": set(memory["blocked"]),
            "marker": memory.get("marker"),
            "radius": int(memory.get("radius", 0)),
        }
    )
    scores = torch.full((len(ACTION_NAMES),), -0.25, dtype=torch.float32)
    scores[ACTION_NAMES.index(teacher_action)] = 1.0
    if (1, 0) in memory["blocked"]:
        scores[ACTION_NAMES.index("forward")] = -1.0
    marker = memory.get("marker")
    if marker == (0, 0):
        scores.fill_(-1.0)
        scores[ACTION_NAMES.index("hold")] = 1.0
    return scores


def _json_safe_arg(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, list):
        return [_json_safe_arg(item) for item in value]
    if isinstance(value, tuple):
        return [_json_safe_arg(item) for item in value]
    return value


def _teacher_scores_from_action(action: str) -> torch.Tensor:
    scores = torch.full((len(ACTION_NAMES),), -0.25, dtype=torch.float32)
    scores[ACTION_NAMES.index(action)] = 1.0
    return scores


def _teacher_scores_from_tensor(memory: torch.Tensor) -> torch.Tensor:
    return _teacher_scores_from_memory(_memory_from_tensor(memory))


def _policy_dataset(
    memories: torch.Tensor,
    actions: torch.Tensor,
) -> TensorDataset:
    return TensorDataset(
        memories,
        actions,
        torch.stack([_teacher_scores_from_tensor(memory) for memory in memories]),
    )


@torch.no_grad()
def _evaluate(
    policy: Phase3AEgocentricMemoryPolicy,
    dataset: TensorDataset,
    *,
    batch_size: int,
    device: torch.device,
) -> dict:
    policy.eval()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    total = 0
    correct = 0
    total_loss = 0.0
    total_value_loss = 0.0
    total_safety_violations = 0
    total_blocked_forward = 0
    action_counts = {name: 0 for name in ACTION_NAMES}
    pred_counts = {name: 0 for name in ACTION_NAMES}
    for memory, action, score_target in loader:
        memory = memory.to(device)
        action = action.to(device)
        score_target = score_target.to(device)
        logits = policy(memory)
        loss = F.cross_entropy(logits, action)
        value_loss = F.mse_loss(logits, score_target)
        pred = logits.argmax(dim=1)
        correct += int((pred == action).sum().item())
        total += int(action.shape[0])
        total_loss += float(loss.item()) * int(action.shape[0])
        total_value_loss += float(value_loss.item()) * int(action.shape[0])
        radius = memory.shape[-1] // 2
        blocked_forward = memory[:, 0, radius - 1, radius] >= 0.5
        if bool(blocked_forward.any()):
            total_blocked_forward += int(blocked_forward.sum().item())
            total_safety_violations += int(
                (pred[blocked_forward] == ACTION_NAMES.index("forward")).sum().item()
            )
        for action_id, name in enumerate(ACTION_NAMES):
            action_counts[name] += int((action == action_id).sum().item())
            pred_counts[name] += int((pred == action_id).sum().item())
    return {
        "examples": total,
        "loss": total_loss / max(total, 1),
        "value_loss": total_value_loss / max(total, 1),
        "action_match": correct / max(total, 1),
        "blocked_forward_examples": total_blocked_forward,
        "blocked_forward_violation_rate": total_safety_violations
        / max(total_blocked_forward, 1),
        "target_action_counts": action_counts,
        "predicted_action_counts": pred_counts,
    }


def _synthetic_policy_dataset(
    *,
    count: int,
    memory_size: int,
    seed: int,
) -> TensorDataset:
    memories = []
    actions = []
    rng = random.Random(seed)
    for _ in range(count):
        tensor = _random_memory_tensor(rng, memory_size=memory_size)
        memory = _memory_from_tensor(tensor)
        action = _select_egocentric_frontier_action(memory)
        memories.append(tensor)
        actions.append(ACTION_NAMES.index(action))
    if not memories:
        return TensorDataset(
            torch.empty(0, 3, memory_size, memory_size),
            torch.empty(0, dtype=torch.long),
            torch.empty(0, len(ACTION_NAMES), dtype=torch.float32),
        )
    return TensorDataset(
        torch.stack(memories),
        torch.tensor(actions, dtype=torch.long),
        torch.stack([_teacher_scores_from_tensor(memory) for memory in memories]),
    )


def _concat_policy_datasets(first: TensorDataset, second: TensorDataset) -> TensorDataset:
    if len(second.tensors[1]) == 0:
        return first
    return TensorDataset(
        torch.cat([first.tensors[0], second.tensors[0]], dim=0),
        torch.cat([first.tensors[1], second.tensors[1]], dim=0),
        torch.cat([first.tensors[2], second.tensors[2]], dim=0),
    )


def _action_weights(dataset: TensorDataset) -> torch.Tensor:
    actions = dataset.tensors[1]
    counts = torch.bincount(actions, minlength=len(ACTION_NAMES)).to(torch.float32)
    weights = counts.sum() / counts.clamp_min(1.0)
    return weights / weights.mean().clamp_min(1e-6)


def _load_latent_memory_updater(
    path: Path,
    *,
    model: nn.Module,
    device: torch.device,
) -> tuple[Phase3AEgocentricMemoryUpdate, dict]:
    try:
        checkpoint = torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        checkpoint = torch.load(path, map_location=device)
    report = checkpoint.get("report", {})
    config = report.get("model_config", {})
    updater = Phase3AEgocentricMemoryUpdate(
        memory_size=int(config.get("memory_size", model.spatial_memory_size)),
        hidden_dim=int(config.get("hidden_dim", 96)),
        memory_channels=int(config.get("memory_channels", 3)),
        evidence_channels=int(config.get("evidence_channels", 3)),
        action_dim=int(config.get("action_dim", len(ACTION_NAMES))),
        use_geometric_prior=bool(config.get("use_geometric_prior", True)),
        learned_transition_hidden_dim=config.get("learned_transition_hidden_dim"),
    ).to(device)
    updater.load_state_dict(checkpoint["updater_state_dict"])
    updater.eval()
    return updater, report


def _select_policy_teacher_action(
    *,
    teacher_source: str,
    memory_dict: dict,
    odom_frontier_memory: dict,
    scene,
    state,
    view_size: int,
    odom_lookahead_horizon: int,
    odom_lookahead_beam_width: int,
) -> str:
    if teacher_source == "egocentric_frontier":
        return _select_egocentric_frontier_action(memory_dict)
    if teacher_source == "odom_frontier":
        _update_odom_frontier_memory(
            odom_frontier_memory,
            scene=scene,
            state=state,
            view_size=view_size,
            current_goal_marker=True,
        )
        return _select_odom_frontier_action(odom_frontier_memory, state)
    if teacher_source == "odom_lookahead":
        _update_odom_frontier_memory(
            odom_frontier_memory,
            scene=scene,
            state=state,
            view_size=view_size,
            current_goal_marker=True,
        )
        return _select_odom_frontier_lookahead_action(
            odom_frontier_memory,
            scene=scene,
            state=state,
            view_size=view_size,
            horizon=odom_lookahead_horizon,
            beam_width=odom_lookahead_beam_width,
        )
    raise ValueError(f"unknown policy teacher source: {teacher_source!r}")


@torch.no_grad()
def _collect_dagger_dataset(
    policy: Phase3AEgocentricMemoryPolicy,
    rows: list[dict],
    *,
    scene_seed: int,
    width: int,
    height: int,
    view_size: int,
    memory_size: int,
    max_episodes: int | None,
    max_steps: int,
    base_model: nn.Module,
    latent_map_head,
    latent_memory_updater: Phase3AEgocentricMemoryUpdate,
    blocked_threshold: float,
    free_threshold: float,
    marker_threshold: float,
    teacher_source: str,
    odom_lookahead_horizon: int,
    odom_lookahead_beam_width: int,
    student_action_probability: float,
    random_action_probability: float,
    device: torch.device,
) -> TensorDataset:
    grouped = {}
    for row in rows:
        grouped.setdefault((str(row["scene_id"]), int(row["source_index"])), []).append(row)
    groups = [grouped[key] for key in sorted(grouped)]
    if max_episodes is not None:
        groups = groups[:max_episodes]
    memories = []
    actions = []
    policy.eval()
    rng = random.Random(20260648 + len(groups) + max_steps)
    for group_index, group in enumerate(groups):
        template = group[0]
        scene = _goal_scene_from_row(template, seed=scene_seed, width=width, height=height)
        state = _state_from_dict(template["start_state"])
        recurrent_memory = torch.zeros(
            1,
            3,
            memory_size,
            memory_size,
            dtype=torch.float32,
            device=device,
        )
        last_action = "hold"
        last_collision = False
        odom_frontier_memory = {
            "free": set(),
            "blocked": set(),
            "marker": None,
            "radius": max(memory_size // 2, 0),
        }
        for _step in range(max_steps):
            local_evidence = _latent_local_evidence(
                model=base_model,
                latent_map_head=latent_map_head,
                scene=scene,
                state=state,
                view_size=view_size,
                current_goal_marker=True,
                device=device,
            )
            local_evidence = _center_local_evidence(
                local_evidence,
                memory_size=memory_size,
            )
            logits = latent_memory_updater(
                recurrent_memory,
                local_evidence,
                _action_index_tensor(last_action, device=device),
                torch.tensor([float(last_collision)], dtype=torch.float32, device=device),
            )
            recurrent_memory = logits.sigmoid().detach()
            memory_dict = _egocentric_memory_tensor_to_dict(
                recurrent_memory[0].detach().cpu(),
                blocked_threshold=blocked_threshold,
                free_threshold=free_threshold,
                marker_threshold=marker_threshold,
            )
            teacher_action = _select_policy_teacher_action(
                teacher_source=teacher_source,
                memory_dict=memory_dict,
                odom_frontier_memory=odom_frontier_memory,
                scene=scene,
                state=state,
                view_size=view_size,
                odom_lookahead_horizon=odom_lookahead_horizon,
                odom_lookahead_beam_width=odom_lookahead_beam_width,
            )
            memories.append(recurrent_memory[0].detach().cpu())
            actions.append(ACTION_NAMES.index(teacher_action))
            policy_action = ACTION_NAMES[
                int(policy(recurrent_memory).argmax(dim=1).item())
            ]
            draw = rng.random()
            if draw < random_action_probability:
                action = rng.choice(ACTION_NAMES)
            elif draw < random_action_probability + student_action_probability:
                action = policy_action
            else:
                action = teacher_action
            next_state, collision = step_state(scene, state, action)
            last_action = action
            last_collision = bool(collision)
            state = next_state
            if (state.x, state.y) == scene.goal:
                break
        if (group_index + 1) % 32 == 0:
            print(
                json.dumps(
                    {
                        "dagger_groups": group_index + 1,
                        "dagger_examples": len(actions),
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
    if not memories:
        return TensorDataset(
            torch.empty(0, 3, memory_size, memory_size),
            torch.empty(0, dtype=torch.long),
            torch.empty(0, len(ACTION_NAMES), dtype=torch.float32),
        )
    memory_tensor = torch.stack(memories)
    return TensorDataset(
        memory_tensor,
        torch.tensor(actions, dtype=torch.long),
        torch.stack(
            [_teacher_scores_from_action(ACTION_NAMES[int(action)]) for action in actions]
        ),
    )


@torch.no_grad()
def _collect_trace_policy_dataset(
    trace_paths: list[Path],
    rows: list[dict],
    *,
    scene_seed: int,
    width: int,
    height: int,
    view_size: int,
    memory_size: int,
    base_model: nn.Module,
    latent_map_head,
    latent_memory_updater: Phase3AEgocentricMemoryUpdate,
    blocked_threshold: float,
    free_threshold: float,
    marker_threshold: float,
    teacher_source: str,
    odom_lookahead_horizon: int,
    odom_lookahead_beam_width: int,
    failed_only: bool,
    no_reachable_marker_only: bool,
    repeat: int,
    device: torch.device,
) -> TensorDataset:
    grouped = {}
    for row in rows:
        grouped.setdefault((str(row["scene_id"]), int(row["source_index"])), []).append(row)
    groups = [grouped[key] for key in sorted(grouped)]
    memories = []
    actions = []
    for trace_path in trace_paths:
        trace = json.loads(trace_path.read_text())
        for episode_index, episode in enumerate(trace.get("episodes", [])):
            if episode_index >= len(groups):
                raise SystemExit(
                    f"{trace_path} has episode index {episode_index}, "
                    f"but only {len(groups)} source groups are available"
                )
            if failed_only and bool(episode.get("claimed", False)):
                continue
            template = groups[episode_index][0]
            scene = _goal_scene_from_row(
                template,
                seed=scene_seed,
                width=width,
                height=height,
            )
            recurrent_memory = torch.zeros(
                1,
                3,
                memory_size,
                memory_size,
                dtype=torch.float32,
                device=device,
            )
            odom_frontier_memory = {
                "free": set(),
                "blocked": set(),
                "marker": None,
                "radius": max(memory_size // 2, 0),
            }
            last_action = "hold"
            last_collision = False
            for item in episode.get("trajectory", []):
                state = _state_from_dict(item["state"])
                local_evidence = _latent_local_evidence(
                    model=base_model,
                    latent_map_head=latent_map_head,
                    scene=scene,
                    state=state,
                    view_size=view_size,
                    current_goal_marker=True,
                    device=device,
                )
                local_evidence = _center_local_evidence(
                    local_evidence,
                    memory_size=memory_size,
                )
                logits = latent_memory_updater(
                    recurrent_memory,
                    local_evidence,
                    _action_index_tensor(last_action, device=device),
                    torch.tensor(
                        [float(last_collision)],
                        dtype=torch.float32,
                        device=device,
                    ),
                )
                recurrent_memory = logits.sigmoid().detach()
                memory_dict = _egocentric_memory_tensor_to_dict(
                    recurrent_memory[0].detach().cpu(),
                    blocked_threshold=blocked_threshold,
                    free_threshold=free_threshold,
                    marker_threshold=marker_threshold,
                )
                marker = memory_dict.get("marker")
                if (
                    no_reachable_marker_only
                    and marker is not None
                    and marker in memory_dict["free"]
                ):
                    last_action = str(item["selected_action"])
                    last_collision = bool(item["collision"])
                    continue
                teacher_action = _select_policy_teacher_action(
                    teacher_source=teacher_source,
                    memory_dict=memory_dict,
                    odom_frontier_memory=odom_frontier_memory,
                    scene=scene,
                    state=state,
                    view_size=view_size,
                    odom_lookahead_horizon=odom_lookahead_horizon,
                    odom_lookahead_beam_width=odom_lookahead_beam_width,
                )
                for _ in range(repeat):
                    memories.append(recurrent_memory[0].detach().cpu())
                    actions.append(ACTION_NAMES.index(teacher_action))
                last_action = str(item["selected_action"])
                last_collision = bool(item["collision"])
    if not memories:
        return TensorDataset(
            torch.empty(0, 3, memory_size, memory_size),
            torch.empty(0, dtype=torch.long),
            torch.empty(0, len(ACTION_NAMES), dtype=torch.float32),
        )
    return TensorDataset(
        torch.stack(memories),
        torch.tensor(actions, dtype=torch.long),
        torch.stack(
            [_teacher_scores_from_action(ACTION_NAMES[int(action)]) for action in actions]
        ),
    )


def _training_loss(
    logits: torch.Tensor,
    action: torch.Tensor,
    score_target: torch.Tensor,
    memory: torch.Tensor,
    *,
    class_weights: torch.Tensor | None,
    value_loss_weight: float,
    safety_loss_weight: float,
) -> torch.Tensor:
    loss = F.cross_entropy(logits, action, weight=class_weights)
    if value_loss_weight > 0.0:
        loss = loss + value_loss_weight * F.mse_loss(logits, score_target)
    if safety_loss_weight > 0.0:
        radius = memory.shape[-1] // 2
        blocked_forward = memory[:, 0, radius - 1, radius] >= 0.5
        if bool(blocked_forward.any()):
            forward_logits = logits[blocked_forward, ACTION_NAMES.index("forward")]
            loss = loss + safety_loss_weight * F.softplus(forward_logits).mean()
    return loss


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-data", type=Path, required=True)
    parser.add_argument("--validation-data", type=Path, required=True)
    parser.add_argument("--base-checkpoint", type=Path, required=True)
    parser.add_argument("--latent-map-head", type=Path, required=True)
    parser.add_argument("--latent-memory-updater", type=Path, default=None)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--width-cells", type=int, default=17)
    parser.add_argument("--height-cells", type=int, default=17)
    parser.add_argument("--view-size", type=int, default=7)
    parser.add_argument("--memory-size", type=int, default=31)
    parser.add_argument("--max-train-episodes", type=int, default=None)
    parser.add_argument("--max-validation-episodes", type=int, default=16)
    parser.add_argument("--max-steps", type=int, default=68)
    parser.add_argument("--optimization-steps", type=int, default=2048)
    parser.add_argument("--dagger-rounds", type=int, default=0)
    parser.add_argument("--dagger-steps-per-round", type=int, default=1024)
    parser.add_argument("--dagger-max-train-episodes", type=int, default=None)
    parser.add_argument("--dagger-student-action-probability", type=float, default=1.0)
    parser.add_argument("--dagger-random-action-probability", type=float, default=0.0)
    parser.add_argument(
        "--teacher-source",
        choices=("egocentric_frontier", "odom_frontier", "odom_lookahead"),
        default="egocentric_frontier",
    )
    parser.add_argument("--odom-lookahead-horizon", type=int, default=9)
    parser.add_argument("--odom-lookahead-beam-width", type=int, default=32)
    parser.add_argument("--trace-policy-data", type=Path, action="append", default=[])
    parser.add_argument(
        "--trace-policy-source-data",
        type=Path,
        action="append",
        default=[],
    )
    parser.add_argument("--trace-policy-failed-only", action="store_true")
    parser.add_argument("--trace-policy-no-reachable-marker-only", action="store_true")
    parser.add_argument("--trace-policy-repeat", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--hidden-dim", type=int, default=512)
    parser.add_argument("--architecture", choices=("mlp", "conv"), default="mlp")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--latent-map-blocked-threshold", type=float, default=0.5)
    parser.add_argument("--latent-map-marker-threshold", type=float, default=0.9)
    parser.add_argument("--latent-memory-blocked-threshold", type=float, default=0.5)
    parser.add_argument("--latent-memory-free-threshold", type=float, default=0.5)
    parser.add_argument("--latent-memory-marker-threshold", type=float, default=0.9)
    parser.add_argument("--include-marker-start-train-groups", action="store_true")
    parser.add_argument("--synthetic-policy-examples", type=int, default=0)
    parser.add_argument("--balanced-action-loss", action="store_true")
    parser.add_argument("--value-loss-weight", type=float, default=0.0)
    parser.add_argument("--safety-loss-weight", type=float, default=0.0)
    parser.add_argument("--save-best-action-match", action="store_true")
    parser.add_argument("--seed", type=int, default=20260647)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--log-every", type=int, default=256)
    args = parser.parse_args()

    if args.memory_size < args.view_size:
        raise SystemExit("--memory-size must be >= --view-size")
    if args.memory_size % 2 == 0:
        raise SystemExit("--memory-size must be odd")
    if args.optimization_steps < 1:
        raise SystemExit("--optimization-steps must be positive")
    if args.dagger_rounds < 0:
        raise SystemExit("--dagger-rounds must be non-negative")
    if args.dagger_rounds > 0 and args.latent_memory_updater is None:
        raise SystemExit("--dagger-rounds requires --latent-memory-updater")
    if args.trace_policy_data and args.latent_memory_updater is None:
        raise SystemExit("--trace-policy-data requires --latent-memory-updater")
    if args.odom_lookahead_horizon < 1:
        raise SystemExit("--odom-lookahead-horizon must be positive")
    if args.odom_lookahead_beam_width < 1:
        raise SystemExit("--odom-lookahead-beam-width must be positive")
    if args.trace_policy_repeat < 1:
        raise SystemExit("--trace-policy-repeat must be positive")
    if args.trace_policy_source_data and (
        len(args.trace_policy_source_data) != len(args.trace_policy_data)
    ):
        raise SystemExit(
            "--trace-policy-source-data must be passed once for each "
            "--trace-policy-data path"
        )
    if args.dagger_steps_per_round < 1:
        raise SystemExit("--dagger-steps-per-round must be positive")
    if (
        args.dagger_max_train_episodes is not None
        and args.dagger_max_train_episodes < 1
    ):
        raise SystemExit("--dagger-max-train-episodes must be positive")
    if not 0.0 <= args.dagger_student_action_probability <= 1.0:
        raise SystemExit("--dagger-student-action-probability must be in [0, 1]")
    if not 0.0 <= args.dagger_random_action_probability <= 1.0:
        raise SystemExit("--dagger-random-action-probability must be in [0, 1]")
    if args.dagger_student_action_probability + args.dagger_random_action_probability > 1.0:
        raise SystemExit(
            "--dagger-student-action-probability + "
            "--dagger-random-action-probability must be <= 1"
        )
    if args.hidden_dim < 1:
        raise SystemExit("--hidden-dim must be positive")
    if args.synthetic_policy_examples < 0:
        raise SystemExit("--synthetic-policy-examples must be non-negative")
    if args.value_loss_weight < 0.0:
        raise SystemExit("--value-loss-weight must be non-negative")
    if args.safety_loss_weight < 0.0:
        raise SystemExit("--safety-loss-weight must be non-negative")

    torch.manual_seed(args.seed)
    device = torch.device(
        "cuda"
        if args.device == "auto" and torch.cuda.is_available()
        else "cpu"
        if args.device == "auto"
        else args.device
    )
    train_seed = _infer_scene_seed(args.train_data)
    validation_seed = _infer_scene_seed(args.validation_data)
    if train_seed is None or validation_seed is None:
        raise SystemExit("could not infer train/validation scene seeds")
    train_rows = read_jsonl(args.train_data)
    validation_rows = read_jsonl(args.validation_data)
    base_model, base_report = load_model(args.base_checkpoint, device=device)
    base_model.eval()
    for parameter in base_model.parameters():
        parameter.requires_grad_(False)
    latent_map_head, latent_map_report = _load_latent_map_head(
        args.latent_map_head,
        base_model=base_model,
        device=device,
    )
    latent_memory_updater = None
    latent_memory_report = None
    if args.latent_memory_updater is not None:
        latent_memory_updater, latent_memory_report = _load_latent_memory_updater(
            args.latent_memory_updater,
            model=base_model,
            device=device,
        )
        if int(latent_memory_updater.memory_size) != int(args.memory_size):
            raise SystemExit(
                "--memory-size must match latent memory updater size "
                f"({latent_memory_updater.memory_size})"
            )
    train_examples = _build_examples(
        train_rows,
        scene_seed=train_seed,
        width=args.width_cells,
        height=args.height_cells,
        view_size=args.view_size,
        memory_size=args.memory_size,
        max_episodes=args.max_train_episodes,
        max_steps=args.max_steps,
        base_model=base_model,
        latent_map_head=latent_map_head,
        blocked_threshold=args.latent_map_blocked_threshold,
        marker_threshold=args.latent_map_marker_threshold,
        device=device,
        include_marker_start_groups=args.include_marker_start_train_groups,
    )
    validation_examples = _build_examples(
        validation_rows,
        scene_seed=validation_seed,
        width=args.width_cells,
        height=args.height_cells,
        view_size=args.view_size,
        memory_size=args.memory_size,
        max_episodes=args.max_validation_episodes,
        max_steps=args.max_steps,
        base_model=base_model,
        latent_map_head=latent_map_head,
        blocked_threshold=args.latent_map_blocked_threshold,
        marker_threshold=args.latent_map_marker_threshold,
        device=device,
        include_marker_start_groups=False,
    )
    train_dataset = _policy_dataset(
        train_examples.targets,
        train_examples.teacher_actions,
    )
    if args.synthetic_policy_examples > 0:
        train_dataset = _concat_policy_datasets(
            train_dataset,
            _synthetic_policy_dataset(
                count=args.synthetic_policy_examples,
                memory_size=args.memory_size,
                seed=args.seed + 17021,
            ),
        )
    trace_policy_examples = 0
    if args.trace_policy_data:
        assert latent_memory_updater is not None
        if args.trace_policy_source_data:
            for trace_path, source_path in zip(
                args.trace_policy_data,
                args.trace_policy_source_data,
                strict=True,
            ):
                trace_seed = _infer_scene_seed(source_path)
                if trace_seed is None:
                    raise SystemExit(
                        "could not infer trace-policy source scene seed "
                        f"from {source_path}"
                    )
                trace_dataset = _collect_trace_policy_dataset(
                    [trace_path],
                    read_jsonl(source_path),
                    scene_seed=trace_seed,
                    width=args.width_cells,
                    height=args.height_cells,
                    view_size=args.view_size,
                    memory_size=args.memory_size,
                    base_model=base_model,
                    latent_map_head=latent_map_head,
                    latent_memory_updater=latent_memory_updater,
                    blocked_threshold=args.latent_memory_blocked_threshold,
                    free_threshold=args.latent_memory_free_threshold,
                    marker_threshold=args.latent_memory_marker_threshold,
                    teacher_source=args.teacher_source,
                    odom_lookahead_horizon=args.odom_lookahead_horizon,
                    odom_lookahead_beam_width=args.odom_lookahead_beam_width,
                    failed_only=bool(args.trace_policy_failed_only),
                    no_reachable_marker_only=bool(
                        args.trace_policy_no_reachable_marker_only
                    ),
                    repeat=int(args.trace_policy_repeat),
                    device=device,
                )
                trace_policy_examples += int(len(trace_dataset.tensors[1]))
                train_dataset = _concat_policy_datasets(train_dataset, trace_dataset)
        else:
            trace_dataset = _collect_trace_policy_dataset(
                list(args.trace_policy_data),
                validation_rows,
                scene_seed=validation_seed,
                width=args.width_cells,
                height=args.height_cells,
                view_size=args.view_size,
                memory_size=args.memory_size,
                base_model=base_model,
                latent_map_head=latent_map_head,
                latent_memory_updater=latent_memory_updater,
                blocked_threshold=args.latent_memory_blocked_threshold,
                free_threshold=args.latent_memory_free_threshold,
                marker_threshold=args.latent_memory_marker_threshold,
                teacher_source=args.teacher_source,
                odom_lookahead_horizon=args.odom_lookahead_horizon,
                odom_lookahead_beam_width=args.odom_lookahead_beam_width,
                failed_only=bool(args.trace_policy_failed_only),
                no_reachable_marker_only=bool(
                    args.trace_policy_no_reachable_marker_only
                ),
                repeat=int(args.trace_policy_repeat),
                device=device,
            )
            trace_policy_examples = int(len(trace_dataset.tensors[1]))
            train_dataset = _concat_policy_datasets(train_dataset, trace_dataset)
    validation_dataset = _policy_dataset(
        validation_examples.targets,
        validation_examples.teacher_actions,
    )
    policy = Phase3AEgocentricMemoryPolicy(
        memory_size=args.memory_size,
        hidden_dim=args.hidden_dim,
        action_dim=len(ACTION_NAMES),
        architecture=args.architecture,
    ).to(device)
    optimizer = torch.optim.AdamW(
        policy.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    class_weights = (
        _action_weights(train_dataset).to(device)
        if args.balanced_action_loss
        else None
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=False,
    )
    iterator = iter(train_loader)
    logs = []
    best_state = None
    best_metrics = None
    best_step = None
    best_score = (-1.0, float("-inf"))
    total_steps = args.optimization_steps + args.dagger_rounds * args.dagger_steps_per_round
    current_step = 0

    def _train_for_steps(steps: int, *, stage: str) -> None:
        nonlocal iterator
        nonlocal best_metrics
        nonlocal best_score
        nonlocal best_state
        nonlocal best_step
        nonlocal current_step
        for _ in range(steps):
            current_step += 1
            try:
                memory, action, score_target = next(iterator)
            except StopIteration:
                iterator = iter(train_loader)
                memory, action, score_target = next(iterator)
            memory = memory.to(device)
            action = action.to(device)
            score_target = score_target.to(device)
            logits = policy(memory)
            loss = _training_loss(
                logits,
                action,
                score_target,
                memory,
                class_weights=class_weights,
                value_loss_weight=args.value_loss_weight,
                safety_loss_weight=args.safety_loss_weight,
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            if current_step % args.log_every == 0 or current_step == total_steps:
                metrics = _evaluate(
                    policy,
                    validation_dataset,
                    batch_size=args.batch_size,
                    device=device,
                )
                entry = {
                    "step": current_step,
                    "stage": stage,
                    "train_loss": float(loss.item()),
                    **metrics,
                }
                logs.append(entry)
                print(json.dumps(entry, sort_keys=True), flush=True)
                score = (
                    -float(metrics["blocked_forward_violation_rate"]),
                    float(metrics["action_match"]),
                    -float(metrics["loss"]),
                )
                if args.save_best_action_match and score > best_score:
                    best_score = score
                    best_step = current_step
                    best_metrics = dict(metrics)
                    best_state = {
                        key: value.detach().cpu().clone()
                        for key, value in policy.state_dict().items()
                    }

    _train_for_steps(args.optimization_steps, stage="initial")
    for dagger_round in range(1, args.dagger_rounds + 1):
        assert latent_memory_updater is not None
        dagger_dataset = _collect_dagger_dataset(
            policy,
            train_rows,
            scene_seed=train_seed,
            width=args.width_cells,
            height=args.height_cells,
            view_size=args.view_size,
            memory_size=args.memory_size,
            max_episodes=args.dagger_max_train_episodes,
            max_steps=args.max_steps,
            base_model=base_model,
            latent_map_head=latent_map_head,
            latent_memory_updater=latent_memory_updater,
            blocked_threshold=args.latent_memory_blocked_threshold,
            free_threshold=args.latent_memory_free_threshold,
            marker_threshold=args.latent_memory_marker_threshold,
            teacher_source=args.teacher_source,
            odom_lookahead_horizon=args.odom_lookahead_horizon,
            odom_lookahead_beam_width=args.odom_lookahead_beam_width,
            student_action_probability=args.dagger_student_action_probability,
            random_action_probability=args.dagger_random_action_probability,
            device=device,
        )
        train_dataset = _concat_policy_datasets(train_dataset, dagger_dataset)
        class_weights = (
            _action_weights(train_dataset).to(device)
            if args.balanced_action_loss
            else None
        )
        train_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            drop_last=False,
        )
        iterator = iter(train_loader)
        print(
            json.dumps(
                {
                    "dagger_round": dagger_round,
                    "dagger_examples": int(len(dagger_dataset.tensors[1])),
                    "train_total_examples": int(len(train_dataset.tensors[1])),
                },
                sort_keys=True,
            ),
            flush=True,
        )
        _train_for_steps(
            args.dagger_steps_per_round,
            stage=f"dagger_{dagger_round}",
        )

    final_validation = _evaluate(
        policy,
        validation_dataset,
        batch_size=args.batch_size,
        device=device,
    )
    selected_step = total_steps
    selected_validation = final_validation
    if args.save_best_action_match and best_state is not None:
        policy.load_state_dict(best_state)
        selected_step = int(best_step)
        selected_validation = best_metrics or final_validation
    report = {
        "schema": "jepa_phase3a_latent_policy_training_report_v0",
        "base_checkpoint": str(args.base_checkpoint.resolve()),
        "base_checkpoint_completed_steps": base_report.get("completed_steps"),
        "latent_map_head": str(args.latent_map_head.resolve()),
        "latent_map_completed_steps": latent_map_report.get("completed_steps"),
        "latent_memory_updater": (
            str(args.latent_memory_updater.resolve())
            if args.latent_memory_updater is not None
            else None
        ),
        "latent_memory_completed_steps": (
            latent_memory_report.get("completed_steps")
            if latent_memory_report is not None
            else None
        ),
        "train_data": str(args.train_data.resolve()),
        "validation_data": str(args.validation_data.resolve()),
        "train_seed": train_seed,
        "validation_seed": validation_seed,
        "train_examples": int(len(train_examples.teacher_actions)),
        "trace_policy_examples": int(trace_policy_examples),
        "train_total_examples": int(len(train_dataset.tensors[1])),
        "validation_examples": int(len(validation_examples.teacher_actions)),
        "completed_steps": total_steps,
        "final_validation": final_validation,
        "selected_step": selected_step,
        "selected_validation": selected_validation,
        "logs": logs,
        "args": {
            key: _json_safe_arg(value)
            for key, value in vars(args).items()
        },
        "model_config": {
            "memory_size": args.memory_size,
            "hidden_dim": args.hidden_dim,
            "memory_channels": 3,
            "action_dim": len(ACTION_NAMES),
            "architecture": args.architecture,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "policy_state_dict": policy.state_dict(),
            "report": report,
        },
        args.output,
    )
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
