#!/usr/bin/env python3
"""Export an MP4 visualization for a Phase 3A memory positive-control run."""
from __future__ import annotations

import argparse
import math
import sys
from collections import defaultdict
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from lewm.benchmarks.phase3a_positive_control import ACTION_NAMES, read_jsonl  # noqa: E402
from lewm.benchmarks.phase3a_explore_claim import egocentric_explore_claim_score  # noqa: E402
from lewm.benchmarks.phase3a_marker_memory import (  # noqa: E402
    egocentric_marker_memory_score,
)
from lewm.benchmarks.phase3a_training import (  # noqa: E402
    Phase3AMaterializedDataset,
    materialize_phase3a_batch,
    source_key,
)
from lewm.models.phase3a_jepa import Phase3AJepaModel  # noqa: E402


MODEL_CONFIG_KEYS = {
    "view_size",
    "spatial_memory_size",
    "latent_dim",
    "action_dim",
    "pred_layers",
    "target_ema_momentum",
    "prediction_loss_lambda",
    "action_identifiability_lambda",
    "zero_action_lambda",
    "free_running_action_contrast_lambda",
    "free_running_zero_contrast_lambda",
    "utility_loss_lambda",
    "utility_ranking_loss_lambda",
    "utility_ranking_regression_weight",
    "utility_ranking_loss_type",
    "utility_softmax_temperature",
    "utility_source",
    "candidate_score_loss_lambda",
    "candidate_score_regression_weight",
    "candidate_score_ranking_loss_type",
    "candidate_score_softmax_temperature",
    "detach_candidate_score_state",
    "candidate_score_gradient_mode",
    "candidate_score_source_tokens",
    "candidate_score_action_summary",
    "candidate_claim_loss_lambda",
    "candidate_score_claim_logit_weight",
    "online_marker_memory_score_weight",
    "candidate_marker_memory_loss_lambda",
    "candidate_marker_memory_score_weight",
    "candidate_marker_memory_delta_loss_weight",
    "candidate_marker_memory_claim_loss_weight",
    "candidate_marker_memory_ranking_loss_lambda",
    "candidate_marker_memory_ranking_loss_type",
    "candidate_marker_memory_softmax_temperature",
    "candidate_marker_memory_score_mode",
    "structured_marker_memory_loss_lambda",
    "structured_marker_memory_score_weight",
    "structured_marker_memory_ranking_loss_lambda",
    "structured_marker_memory_softmax_temperature",
    "categorical_marker_memory_loss_lambda",
    "categorical_marker_memory_score_weight",
    "categorical_marker_memory_ranking_loss_lambda",
    "categorical_marker_memory_softmax_temperature",
    "categorical_marker_memory_radius",
    "spatial_marker_memory_loss_lambda",
    "spatial_marker_memory_score_weight",
    "spatial_marker_memory_ranking_loss_lambda",
    "spatial_marker_memory_softmax_temperature",
    "spatial_marker_memory_score_temperature",
    "spatial_frontier_memory_loss_lambda",
    "spatial_frontier_observation_loss_lambda",
    "spatial_frontier_memory_score_loss_lambda",
    "spatial_frontier_memory_score_weight",
    "spatial_frontier_memory_ranking_loss_lambda",
    "spatial_frontier_memory_softmax_temperature",
    "spatial_frontier_memory_occupancy_loss_weight",
    "spatial_frontier_memory_marker_loss_weight",
    "spatial_frontier_memory_marker_cell_loss_weight",
    "spatial_frontier_memory_marker_mass_loss_weight",
    "spatial_frontier_memory_detector_init",
    "spatial_frontier_memory_detector_arch",
    "spatial_frontier_memory_gate_mode",
    "spatial_frontier_marker_source",
    "spatial_frontier_collision_penalty",
    "spatial_frontier_novelty_reward",
    "spatial_frontier_marker_gate_threshold",
    "spatial_frontier_marker_gate_width",
    "spatial_frontier_marker_update_threshold",
    "spatial_frontier_marker_update_width",
    "detach_consequence_head_state",
    "consequence_loss_lambda",
    "rollout_delta_loss_lambda",
    "teacher_forced_delta_loss_lambda",
    "decision_token_count",
    "decision_rollout_mode",
    "decision_recurrent_update",
    "decision_target_geometry",
    "decision_target_scale",
    "decision_prediction_loss_lambda",
    "decision_delta_loss_lambda",
    "decision_teacher_forced_prediction_loss_lambda",
    "decision_teacher_forced_delta_loss_lambda",
    "decision_teacher_forced_action_contrast_lambda",
    "decision_teacher_forced_zero_contrast_lambda",
    "decision_action_contrast_lambda",
    "decision_zero_contrast_lambda",
    "use_memory_context",
    "memory_frame_summary",
    "memory_marker_features",
    "spatial_variance_lambda",
}

ACTION_SHORT = {
    "forward": "fwd",
    "turn_left": "left",
    "turn_right": "right",
    "hold": "hold",
}

SCORE_SOURCE_LABELS = {
    "utility": "Learned JEPA utility",
    "candidate_marker_memory_score": "Learned marker-memory score",
    "candidate_marker_memory_distance_score": "Learned marker-memory distance",
    "online_marker_memory_score": "Online RGB+odometry marker memory",
    "structured_marker_memory_score": "Structured learned marker memory",
    "categorical_marker_memory_score": "Categorical learned marker memory",
    "spatial_marker_memory_score": "Spatial belief-map marker memory",
    "spatial_frontier_memory_score": "Spatial frontier+marker memory",
    "egocentric_marker_memory": "Egocentric marker memory",
    "online_frontier_marker": "Online frontier + marker memory",
}


def _load_font(size: int, *, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    filename = "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"
    for path in (
        Path("/usr/share/fonts/truetype/dejavu") / filename,
        Path("/usr/share/fonts/dejavu") / filename,
    ):
        if path.exists():
            return ImageFont.truetype(str(path), size)
    return ImageFont.load_default()


def _load_checkpoint(path: Path) -> dict:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _load_model(checkpoint: dict) -> Phase3AJepaModel:
    report = checkpoint["report"]
    config = report["model_config"]
    kwargs = {
        key: value
        for key, value in config.items()
        if key in MODEL_CONFIG_KEYS and value is not None
    }
    model = Phase3AJepaModel(**kwargs)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model


def _rgb_image(array: list, *, size: int) -> Image.Image:
    tensor = np.asarray(array, dtype=np.float32)
    if tensor.ndim != 3 or tensor.shape[0] != 3:
        raise ValueError(f"expected channel-major RGB observation, got {tensor.shape}")
    rgb = np.clip(np.moveaxis(tensor, 0, -1) * 255.0, 0, 255).astype(np.uint8)
    image = Image.fromarray(rgb)
    return image.resize((size, size), resample=Image.Resampling.NEAREST)


@torch.no_grad()
def _score_source(
    model: Phase3AJepaModel | None,
    rows: list[dict],
    indices: list[int],
    *,
    score_source: str,
    row_cache: Phase3AMaterializedDataset | None = None,
) -> dict:
    if score_source == "egocentric_marker_memory":
        predictions = [
            float(egocentric_marker_memory_score(rows[row_index]))
            for row_index in indices
        ]
    elif score_source == "online_frontier_marker":
        predictions = [
            float(egocentric_explore_claim_score(rows[row_index]))
            for row_index in indices
        ]
    else:
        if model is None:
            raise ValueError(f"score source {score_source} requires --checkpoint")
        batch = (
            row_cache.materialize_batch(indices)
            if row_cache is not None
            else materialize_phase3a_batch(rows, indices)
        )
        output = model(
            vision=batch.vision,
            history_vision=batch.history_vision,
            history_actions=batch.history_actions,
            actions=batch.actions,
            utility_targets=batch.utility_targets,
            consequence_targets=batch.consequence_targets,
            candidate_marker_memory_valid_mask=batch.marker_memory_valid_mask,
            candidate_marker_memory_delta_targets=batch.marker_memory_delta_targets,
            candidate_marker_memory_claim_targets=batch.marker_memory_claim_targets,
            candidate_marker_memory_score_targets=batch.marker_memory_score_targets,
            structured_marker_memory_valid_mask=batch.marker_memory_start_valid_mask,
            structured_marker_memory_start_delta_targets=(
                batch.marker_memory_start_delta_targets
            ),
            categorical_marker_memory_valid_mask=(
                batch.marker_memory_start_cell_valid_mask
            ),
            categorical_marker_memory_cell_targets=(
                batch.marker_memory_start_cell_targets
            ),
            utility_group_ids=batch.utility_group_ids,
            utility_mask=batch.utility_mask,
            wrong_actions=batch.wrong_actions,
            wrong_mask=batch.wrong_mask,
            non_hold_mask=batch.non_hold_mask,
            return_latents=True,
        )
        if score_source == "utility":
            predictions = output["utility_prediction"].detach().cpu().tolist()
        elif score_source == "candidate_marker_memory_score":
            predictions = (
                output["candidate_marker_memory_score_prediction"]
                .detach()
                .cpu()
                .tolist()
            )
        elif score_source == "online_marker_memory_score":
            predictions = (
                output["online_marker_memory_score_prediction"]
                .detach()
                .cpu()
                .tolist()
            )
        elif score_source == "candidate_marker_memory_distance_score":
            predictions = (
                output["candidate_marker_memory_delta_prediction"]
                .abs()
                .sum(dim=-1)
                .neg()
                .detach()
                .cpu()
                .tolist()
            )
        elif score_source == "structured_marker_memory_score":
            predictions = (
                output["structured_marker_memory_score_prediction"]
                .detach()
                .cpu()
                .tolist()
            )
        elif score_source == "categorical_marker_memory_score":
            predictions = (
                output["categorical_marker_memory_score_prediction"]
                .detach()
                .cpu()
                .tolist()
            )
        elif score_source == "spatial_marker_memory_score":
            predictions = (
                output["spatial_marker_memory_score_prediction"]
                .detach()
                .cpu()
                .tolist()
            )
        elif score_source == "spatial_frontier_memory_score":
            predictions = (
                output["spatial_frontier_memory_score_prediction"]
                .detach()
                .cpu()
                .tolist()
            )
        else:
            raise ValueError(f"unknown score source: {score_source}")
    row_summaries = []
    for local_index, row_index in enumerate(indices):
        row = rows[row_index]
        row_summaries.append(
            {
                "row_index": row_index,
                "prediction": float(predictions[local_index]),
                "utility": float(row["consequence_labels"]["target_utility"]),
                "sequence": tuple(str(action) for action in row["primitive_sequence"]),
                "row": row,
            }
        )
    selected = max(row_summaries, key=lambda item: item["prediction"])
    oracle = max(row_summaries, key=lambda item: item["utility"])
    by_first = {}
    for action in ACTION_NAMES:
        candidates = [item for item in row_summaries if item["sequence"][0] == action]
        by_first[action] = {
            "best_predicted": max(candidates, key=lambda item: item["prediction"]),
            "best_truth": max(candidates, key=lambda item: item["utility"]),
        }
    return {
        "source_key": source_key(selected["row"]),
        "score_source": score_source,
        "selector_label": SCORE_SOURCE_LABELS[score_source],
        "selected": selected,
        "oracle": oracle,
        "by_first": by_first,
        "match": selected["sequence"][0] == oracle["sequence"][0],
        "regret": oracle["utility"] - by_first[selected["sequence"][0]]["best_truth"]["utility"],
    }


def _summaries(
    rows: list[dict],
    model: Phase3AJepaModel | None,
    examples: int,
    *,
    score_source: str,
) -> list[dict]:
    grouped: dict[tuple[str, int], list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        grouped[source_key(row)].append(index)
    row_cache = Phase3AMaterializedDataset(rows) if model is not None else None
    scored = [
        _score_source(
            model,
            rows,
            indices,
            score_source=score_source,
            row_cache=row_cache,
        )
        for _, indices in sorted(grouped.items())
    ]
    matches = [item for item in scored if item["match"]]
    misses = [item for item in scored if not item["match"]]
    selected = []
    while len(selected) < examples and (matches or misses):
        if matches:
            selected.append(matches.pop(0))
        if len(selected) < examples and misses:
            selected.append(misses.pop(0))
    return selected[:examples]


def _draw_label(
    draw: ImageDraw.ImageDraw,
    xy: tuple[int, int],
    text: str,
    font: ImageFont.ImageFont,
    fill: tuple[int, int, int] = (32, 36, 40),
) -> None:
    draw.text(xy, text, font=font, fill=fill)


def _sequence_text(sequence: tuple[str, ...]) -> str:
    return " -> ".join(ACTION_SHORT.get(action, action) for action in sequence)


def _paste_observation(
    canvas: Image.Image,
    draw: ImageDraw.ImageDraw,
    observation: list,
    *,
    x: int,
    y: int,
    label: str,
    size: int,
    border: tuple[int, int, int] = (52, 64, 84),
) -> None:
    image = _rgb_image(observation, size=size)
    canvas.paste(image, (x, y))
    draw.rectangle((x, y, x + size, y + size), outline=border, width=3)
    font = _load_font(16, bold=True)
    _draw_label(draw, (x, y + size + 8), label, font)


def _draw_candidate_cards(
    draw: ImageDraw.ImageDraw,
    summary: dict,
    *,
    x: int,
    y: int,
    width: int,
) -> None:
    title_font = _load_font(26, bold=True)
    font = _load_font(19)
    small = _load_font(16)
    _draw_label(draw, (x, y), "First-action candidates", title_font)
    card_w = (width - 18) // 2
    card_h = 116
    selected_first = summary["selected"]["sequence"][0]
    oracle_first = summary["oracle"]["sequence"][0]
    for index, action in enumerate(ACTION_NAMES):
        cx = x + (index % 2) * (card_w + 18)
        cy = y + 44 + (index // 2) * (card_h + 16)
        model_mark = action == selected_first
        oracle_mark = action == oracle_first
        if model_mark and oracle_mark:
            fill = (223, 247, 230)
            outline = (36, 132, 74)
        elif model_mark:
            fill = (255, 244, 218)
            outline = (173, 117, 20)
        elif oracle_mark:
            fill = (226, 239, 255)
            outline = (47, 94, 160)
        else:
            fill = (246, 248, 250)
            outline = (160, 168, 176)
        draw.rounded_rectangle((cx, cy, cx + card_w, cy + card_h), radius=8, fill=fill, outline=outline, width=3)
        pred = summary["by_first"][action]["best_predicted"]
        truth = summary["by_first"][action]["best_truth"]
        _draw_label(draw, (cx + 14, cy + 10), action, font, fill=(16, 24, 32))
        tags = []
        if model_mark:
            tags.append("selected")
        if oracle_mark:
            tags.append("oracle")
        _draw_label(draw, (cx + 14, cy + 39), " / ".join(tags), small, fill=outline)
        _draw_label(draw, (cx + 14, cy + 64), f"score {pred['prediction']:+.2f}", small)
        _draw_label(draw, (cx + 14, cy + 88), f"best utility {truth['utility']:+.1f}", small)


def _draw_rollout_strip(
    canvas: Image.Image,
    draw: ImageDraw.ImageDraw,
    item: dict,
    *,
    x: int,
    y: int,
    label: str,
    size: int,
    border: tuple[int, int, int],
) -> None:
    font = _load_font(20, bold=True)
    small = _load_font(16)
    row = item["row"]
    _draw_label(draw, (x, y), label, font, fill=border)
    observations = [row["start_observation_rgb"]]
    observations.extend(frame["observation_rgb"] for frame in row["future_observations"])
    labels = ["current"]
    labels.extend(row["primitive_sequence"])
    for index, observation in enumerate(observations):
        ox = x + index * (size + 18)
        oy = y + 34
        image = _rgb_image(observation, size=size)
        canvas.paste(image, (ox, oy))
        draw.rectangle((ox, oy, ox + size, oy + size), outline=border, width=3)
        _draw_label(draw, (ox, oy + size + 7), labels[index], small)


def _render_slide(summary: dict, *, width: int, height: int, progress: float) -> Image.Image:
    canvas = Image.new("RGB", (width, height), (250, 250, 248))
    draw = ImageDraw.Draw(canvas)
    title = _load_font(32, bold=True)
    subtitle = _load_font(21)
    small = _load_font(17)
    source = summary["source_key"]
    selected = summary["selected"]
    oracle = summary["oracle"]
    match = summary["match"]
    status = "MATCH" if match else "MISS"
    status_color = (33, 128, 72) if match else (178, 93, 20)

    _draw_label(draw, (32, 24), "Phase 3A JEPA Memory Demo", title)
    _draw_label(
        draw,
        (32, 68),
        f"{summary['selector_label']} | source: {source[0]} / {source[1]} | current beacon hidden",
        _load_font(19),
    )
    draw.rounded_rectangle((1010, 24, 1248, 78), radius=8, fill=(255, 255, 255), outline=status_color, width=3)
    _draw_label(draw, (1028, 37), f"{status}  regret {summary['regret']:.2f}", _load_font(20), fill=status_color)

    row = selected["row"]
    x0 = 32
    y0 = 136
    frame_size = 78
    _draw_label(draw, (x0, y0 - 34), "Observation history", subtitle)
    for index, observation in enumerate(row.get("history_observations_rgb", [])):
        action = ACTION_SHORT.get(row["history_primitive_sequence"][index], row["history_primitive_sequence"][index])
        label = f"h{index} {action}"
        _paste_observation(
            canvas,
            draw,
            observation,
            x=x0 + index * (frame_size + 24),
            y=y0,
            label=label,
            size=frame_size,
            border=(80, 112, 170),
        )
    current_x = x0 + 4 * (frame_size + 24) + 26
    _paste_observation(
        canvas,
        draw,
        row["start_observation_rgb"],
        x=current_x,
        y=y0,
        label="current",
        size=frame_size,
        border=(32, 36, 40),
    )

    _draw_candidate_cards(draw, summary, x=690, y=108, width=540)

    _draw_label(
        draw,
        (32, 316),
        f"Selected: {_sequence_text(selected['sequence'])}",
        _load_font(22, bold=True),
        fill=(42, 107, 62),
    )
    _draw_label(
        draw,
        (32, 346),
        f"score {selected['prediction']:+.2f}",
        _load_font(19),
        fill=(42, 107, 62),
    )
    _draw_label(
        draw,
        (350, 316),
        f"Oracle: {_sequence_text(oracle['sequence'])}",
        _load_font(22, bold=True),
        fill=(45, 89, 150),
    )
    _draw_label(draw, (350, 346), f"utility {oracle['utility']:+.1f}", _load_font(19), fill=(45, 89, 150))
    _draw_rollout_strip(
        canvas,
        draw,
        selected,
        x=32,
        y=388,
        label="Model-selected rollout",
        size=92,
        border=(42, 107, 62),
    )
    _draw_rollout_strip(
        canvas,
        draw,
        oracle,
        x=456,
        y=388,
        label="Oracle-best rollout",
        size=92,
        border=(45, 89, 150),
    )

    _draw_label(
        draw,
        (32, 652),
        "Controlled 2D positive-control visualization. This is not a Go2 robot demo.",
        small,
        fill=(90, 96, 104),
    )
    bar_x0, bar_y0, bar_x1, bar_y1 = 32, 680, 1248, 696
    draw.rounded_rectangle((bar_x0, bar_y0, bar_x1, bar_y1), radius=8, fill=(226, 229, 233))
    fill_x = bar_x0 + int((bar_x1 - bar_x0) * max(0.0, min(progress, 1.0)))
    draw.rounded_rectangle((bar_x0, bar_y0, fill_x, bar_y1), radius=8, fill=(54, 111, 176))
    return canvas


def export_mp4(
    summaries: list[dict],
    output: Path,
    *,
    fps: int,
    seconds_per_example: float,
    width: int,
    height: int,
) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    frames_per_example = max(int(round(fps * seconds_per_example)), 1)
    with imageio.get_writer(
        str(output),
        fps=fps,
        codec="libx264",
        quality=8,
        macro_block_size=16,
    ) as writer:
        total_frames = max(len(summaries) * frames_per_example, 1)
        frame_index = 0
        for summary in summaries:
            for _ in range(frames_per_example):
                progress = frame_index / max(total_frames - 1, 1)
                frame = _render_slide(summary, width=width, height=height, progress=progress)
                writer.append_data(np.asarray(frame))
                frame_index += 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--validation-data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--examples", type=int, default=8)
    parser.add_argument(
        "--score-source",
        choices=tuple(SCORE_SOURCE_LABELS),
        default="utility",
        help="Which score selects the visualized candidate sequence.",
    )
    parser.add_argument("--fps", type=int, default=12)
    parser.add_argument("--seconds-per-example", type=float, default=2.5)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    args = parser.parse_args()

    if args.examples < 1:
        raise SystemExit("--examples must be positive")
    if args.fps < 1:
        raise SystemExit("--fps must be positive")
    model = None
    if args.checkpoint is not None:
        checkpoint = _load_checkpoint(args.checkpoint)
        model = _load_model(checkpoint)
    elif args.score_source not in ("egocentric_marker_memory", "online_frontier_marker"):
        raise SystemExit(f"--checkpoint is required for score source {args.score_source}")
    rows = read_jsonl(args.validation_data)
    summaries = _summaries(
        rows,
        model,
        args.examples,
        score_source=args.score_source,
    )
    export_mp4(
        summaries,
        args.output,
        fps=args.fps,
        seconds_per_example=args.seconds_per_example,
        width=args.width,
        height=args.height,
    )
    matches = sum(int(item["match"]) for item in summaries)
    print(
        f"wrote {args.output} with {len(summaries)} examples "
        f"({matches} selected/oracle first-action matches; "
        f"score_source={args.score_source})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
