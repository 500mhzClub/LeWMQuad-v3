#!/usr/bin/env python3
"""Bounded exploratory frozen ViT-L place-head experiment.

The default path is training-only.  It reconstructs a small, deterministic
inventory from existing train labels/rendered frames, runs a real-feature
smoke, then trains one 30-epoch head.  Held-out evaluation is intentionally a
separate explicit stage and is not entered by this script automatically.
"""
from __future__ import annotations

import hashlib
import json
import os
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "lewm_worlds"))

from dev_frozen_dense_representation_encoders_v1 import VJepa21CroppedV03Arm
from lewm_worlds.manifest import parse_scene_manifest_dict
from lewm_worlds.scene_graph import SceneGraph
from probe_lewm_latent_aliasing import _find_manifest, _iter_label_files, _load_observations

SEED = 2026081801
FAMILIES = ("large_enclosed_maze", "local_composite_motifs", "loop_alias_stress",
            "medium_enclosed_maze", "open_obstacle_field", "rough_local_dynamics",
            "small_enclosed_maze", "visual_sensor_stress")
ENCODER_SHA256 = "7ea9b7cb4a75d10644a8a8d42cff9e177b10dca8f02173f0eaf2b0bed82838c6"
TEMPERATURE = 0.07
MAX_NODES_PER_SCENE = 4
HISTORY = 8


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def digest(value) -> str:
    raw = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()
    return hashlib.sha256(raw).hexdigest()


class PlaceHead(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.token_norm = torch.nn.LayerNorm(1024)
        self.mlp = torch.nn.Sequential(torch.nn.Linear(1024, 256), torch.nn.GELU(), torch.nn.Linear(256, 128))
        self.proj = torch.nn.Linear(256, 128)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        x = self.mlp(self.token_norm(tokens))
        x = torch.cat((x.mean(dim=-2), x.amax(dim=-2)), dim=-1)
        return torch.nn.functional.normalize(self.proj(x), dim=-1)


def _scene_records(rollout_root: Path, corpus_root: Path):
    by_family = defaultdict(list)
    for fam, path in _iter_label_files(rollout_root, "train", None):
        by_family[fam].append(path)
    chosen = []
    for fam in FAMILIES:
        paths = sorted(by_family.get(fam, []), key=lambda p: p.parent.name)
        if not paths:
            continue
        chosen.append((fam, paths[0]))
    return chosen


def _inventory(rollout_root: Path, render_root: Path, corpus_root: Path):
    records = []
    for family, label_file in _scene_records(rollout_root, corpus_root):
        scene = label_file.parent.name
        manifest_path = _find_manifest(corpus_root, "train", family, scene)
        if manifest_path is None:
            continue
        manifest = parse_scene_manifest_dict(json.loads(manifest_path.read_text()))
        graph = SceneGraph(manifest)
        graph_cells = {int(n.node_id) for n in manifest.graph_nodes}
        by_env = _load_observations(label_file)
        summary = label_file.parents[1] / "rollout" / scene / "summary.json"
        n_envs = int(json.loads(summary.read_text()).get("n_envs", len(by_env))) if summary.exists() else max(by_env) + 1
        # Pick the environment with the most graph-valid labels, matching the frozen trajectory-bank convention.
        env_idx, obs = max(by_env.items(), key=lambda kv: sum(int(c in graph_cells and y >= 0) for c, y in kv[1]))
        by_label = defaultdict(list)
        for step, (cell, yaw) in enumerate(obs):
            if int(cell) not in graph_cells or int(yaw) < 0 or step < HISTORY - 1:
                continue
            window = tuple(render_root / scene / "rgb" / f"frame_{(s*n_envs+env_idx):06d}_env_{env_idx:02d}.png"
                           for s in range(step - HISTORY + 1, step + 1))
            if all(p.is_file() for p in window):
                by_label[(int(cell), int(yaw))].append((step, window))
        labels = sorted((k for k, v in by_label.items() if len(v) >= 2), key=lambda x: (x[0], x[1]))
        for node_label in labels[:MAX_NODES_PER_SCENE]:
            # two distinct existing views from the node history; no fabrication.
            for _, window in by_label[node_label][:2]:
                records.append({"family": family, "scene_id": scene, "node": list(node_label),
                                "cell": node_label[0], "yaw": node_label[1],
                                "paths": [str(p) for p in window],
                                "graph_neighbors": list(graph.neighbors(node_label[0]))})
    if not records:
        raise RuntimeError("no valid frozen training node histories found")
    return records


def _preprocess(path: Path, arm: VJepa21CroppedV03Arm) -> torch.Tensor:
    return arm.preprocess(path)


def encode_unique(records, arm, device, batch_size=4):
    paths = sorted({p for r in records for p in r["paths"]})
    lookup = {}
    for i in range(0, len(paths), batch_size):
        xs = torch.stack([_preprocess(Path(p), arm) for p in paths[i:i + batch_size]]).to(device)
        with torch.no_grad():
            out = arm.tokens(xs).detach().cpu().to(torch.float16)
        for p, t in zip(paths[i:i + batch_size], out):
            lookup[p] = t
    return lookup


def _loss(head, views, nodes, scenes, cells, graphs):
    z = head(views)
    sim = z @ z.T / TEMPERATURE
    n = len(z)
    eye = torch.eye(n, dtype=torch.bool, device=z.device)
    node_codes = {name: i for i, name in enumerate(sorted(set(nodes)))}
    labels = torch.tensor([node_codes[name] for name in nodes], device=z.device)
    same = labels[:, None].eq(labels[None, :])
    valid = ~eye
    # One-hop pairs are excluded from the denominator; same-node positives remain valid.
    for i in range(n):
        for j in range(n):
            if scenes[i] != scenes[j] or nodes[i] != nodes[j]:
                if scenes[i] == scenes[j] and graphs[i].bfs_distance(int(cells[i]), int(cells[j])) <= 1:
                    valid[i, j] = False
    losses = []
    for i in range(n):
        pos = same[i] & valid[i]
        if not bool(pos.any()):
            continue
        denom = sim[i][valid[i]]
        losses.append(-(sim[i][pos].mean() - torch.logsumexp(denom, dim=0)))
    if not losses:
        raise RuntimeError("contrastive batch has no positive pairs")
    return torch.stack(losses).mean(), z


def main() -> int:
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--output", type=Path, default=Path("/home/andrewknowles/.cache/lewm_go2_temporal_v03/frozen_vitl_place_head_v1"))
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=64)
    args = p.parse_args()
    torch.manual_seed(SEED); np.random.seed(SEED); random.seed(SEED)
    output = args.output; output.mkdir(parents=True, exist_ok=True)
    rollout = ROOT / ".generated/datagen_full/rollout"
    render = ROOT / ".generated/datagen_full/render_textured_v03"
    corpus = ROOT / ".generated/scene_corpus/minimum_tex_20260520T211541Z"
    records = _inventory(rollout, render, corpus)
    contract = {"schema": "frozen_vitl_self_supervised_place_head_v1", "seed": SEED,
                "families": sorted({r["family"] for r in records}), "records": records,
                "encoder_checkpoint": str(VJepa21CroppedV03Arm().checkpoint), "encoder_sha256": ENCODER_SHA256,
                "architecture": {"token_norm": "LayerNorm(1024)", "mlp": [1024, 256, 128],
                                 "aggregate": ["mean", "amax"], "projection": [256, 128], "output": 128},
                "temperature": TEMPERATURE, "epochs": args.epochs, "optimizer": {"name": "AdamW", "lr": 3e-4, "weight_decay": 1e-2}}
    contract["digest"] = digest(contract)
    (output / "training_contract.json").write_text(json.dumps(contract, indent=2) + "\n")
    arm = VJepa21CroppedV03Arm(); ckpt_sha = sha256(Path(arm.checkpoint))
    if ckpt_sha != ENCODER_SHA256: raise RuntimeError("frozen encoder digest mismatch")
    device = torch.device(args.device)
    arm.build(device, torch.float32)
    lookup = encode_unique(records, arm, device)
    # Smoke on real frozen training features before the scientific contract is considered active.
    smoke_records = records[:4]
    smoke_paths = sorted({p for r in smoke_records for p in r["paths"]})
    views = torch.stack([lookup[p].float() for p in smoke_paths]).to(device)
    head = PlaceHead().to(device)
    node_ids = [f"{r['scene_id']}::{r['node']}::{k}" for r in smoke_records for k in range(2)]
    # one view per record window, then duplicate paired records for explicit positives
    smoke_views = torch.stack([lookup[r["paths"][0]].float() for r in smoke_records] + [lookup[r["paths"][1]].float() for r in smoke_records]).to(device)
    smoke_nodes = [f"{r['scene_id']}::{r['node']}" for r in smoke_records] * 2
    smoke_scenes = [r["scene_id"] for r in smoke_records] * 2
    smoke_cells = [r["cell"] for r in smoke_records] * 2
    # Graph objects are reconstructed for the smoke records.
    graphs = []
    for r in smoke_records:
        mf = _find_manifest(corpus, "train", r["family"], r["scene_id"])
        graphs.append(SceneGraph(parse_scene_manifest_dict(json.loads(mf.read_text()))))
    loss, z = _loss(head, smoke_views, smoke_nodes, smoke_scenes, smoke_cells, graphs * 2)
    loss.backward()
    if not torch.isfinite(loss) or any(p.grad is None or not torch.isfinite(p.grad).all() for p in head.parameters()):
        raise RuntimeError("training-only smoke nonfinite or missing gradient")
    if not torch.allclose(z.norm(dim=1), torch.ones(len(z), device=device), atol=2e-5, rtol=2e-5):
        raise RuntimeError("descriptor norm smoke failed")
    smoke = {"status": "PASS", "loss": float(loss.detach().cpu()), "n_views": len(smoke_views),
             "finite_gradients": True, "descriptor_unit_norm": True, "no_heldout_opened": True}
    (output / "smoke.json").write_text(json.dumps(smoke, indent=2) + "\n")
    # Reload smoke checkpoint, then train exactly one exploratory seed.
    torch.save({"model": head.state_dict(), "contract_digest": contract["digest"]}, output / "smoke_checkpoint.pt")
    reloaded = PlaceHead().to(device); reloaded.load_state_dict(torch.load(output / "smoke_checkpoint.pt", map_location=device, weights_only=False)["model"])
    head = reloaded
    opt = torch.optim.AdamW(head.parameters(), lr=3e-4, weight_decay=1e-2)
    all_views = torch.stack([lookup[r["paths"][0]].float() for r in records] + [lookup[r["paths"][1]].float() for r in records]).to(device)
    all_nodes = [f"{r['scene_id']}::{r['node']}" for r in records] * 2
    all_scenes = [r["scene_id"] for r in records] * 2
    all_cells = [r["cell"] for r in records] * 2
    all_graphs = []
    for r in records:
        mf = _find_manifest(corpus, "train", r["family"], r["scene_id"])
        all_graphs.append(SceneGraph(parse_scene_manifest_dict(json.loads(mf.read_text()))))
    # 30 deterministic full-batch epochs; final epoch only is retained.
    trace = []
    t0 = time.time()
    for epoch in range(1, args.epochs + 1):
        opt.zero_grad(set_to_none=True)
        l, z = _loss(head, all_views, all_nodes, all_scenes, all_cells, all_graphs * 2)
        l.backward(); torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0); opt.step()
        trace.append({"epoch": epoch, "loss": float(l.detach().cpu()), "finite": bool(torch.isfinite(l))})
    elapsed = time.time() - t0
    final = output / "place_head_epoch30.pt"; torch.save({"model": head.state_dict(), "contract_digest": contract["digest"], "epoch": args.epochs}, final)
    receipt = {"status": "PASS", "seed": SEED, "epochs": args.epochs, "records": len(records), "unique_frames": len(lookup),
               "wall_seconds": elapsed, "trace": trace, "checkpoint_sha256": sha256(final), "contract_digest": contract["digest"],
               "heldout_opened": False, "predictor_opened": False}
    (output / "training_receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
