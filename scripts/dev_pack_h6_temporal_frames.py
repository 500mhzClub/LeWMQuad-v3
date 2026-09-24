#!/usr/bin/env python3
"""DEVELOPMENT-TIER: pre-decode H6 temporal frames into a uint8 memmap.

The registered temporal runtime decodes PNGs one at a time through an
access-counting safe loader, which is correct for a sealed one-shot run but is
not the right throughput path for a larger development diagnostic.

This packs the model-visible positions (`rgb[0:4]`) for every bound H6 row into
one uint8 array so PNG decoding is removed from the training inner loop.

Bit-exactness: `rectify_h6_rgb_bytes` is PNG -> crop(224x168) -> bilinear resize
to 112x112 -> uint8 -> float32/255 -> ImageNet normalize. We store the uint8 at
the 112x112 stage, so the float tensor is reproduced exactly by
`unpack_frames()` below. A `--verify` pass checks this against the frozen
decoder on a random sample.

NOT citable as scientific evidence; writes only under `.generated/dev/`.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import io
import json
import os
from pathlib import Path
import stat
import sys

import numpy as np
import PIL

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from lewm.datasets import go2_explicit_plan_discounted_successor_state_v27 as h6
from lewm.benchmarks import go2_recurrent_jepa_main_pool_census as h6_census
from lewm.datasets import go2_recurrent_h4_rgb_sequences as h4_v1
from lewm.datasets import go2_recurrent_h4_rgb_sequences_v2 as h4_v2

RGB_ROOT = REPO_ROOT / ".generated/datagen_full/render_textured_v03"
OUTPUT_ROOT = REPO_ROOT / ".generated/dev/h6_temporal_pack"
POSITIONS = (0, 1, 2, 3)  # model-visible only; 4/5/6 stay forbidden
FRAME_BYTES = 112 * 112 * 3
PACK_SCHEMA = "dev_h6_temporal_pack_v3"
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def sha256_file(path: Path) -> str:
    """Hash one regular file without following a final-component symlink."""
    selected = Path(path)
    if selected.is_symlink() or not selected.is_file():
        raise ValueError(f"pack artifact is not a regular non-symlink file: {selected}")
    digest = hashlib.sha256()
    with selected.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def source_binding(path: Path) -> dict:
    selected = Path(path)
    return {
        "path": selected.resolve().relative_to(REPO_ROOT).as_posix(),
        "byte_count": selected.stat().st_size,
        "sha256": sha256_file(selected),
    }


def pack_source_bindings() -> dict[str, dict]:
    """Bind the packer and the complete local H6 validation closure."""

    return {
        "lewm_package": source_binding(REPO_ROOT / "lewm/__init__.py"),
        "benchmarks_package": source_binding(
            REPO_ROOT / "lewm/benchmarks/__init__.py"
        ),
        "counterfactual_metrics": source_binding(
            REPO_ROOT / "lewm/benchmarks/counterfactual.py"
        ),
        "datasets_package": source_binding(REPO_ROOT / "lewm/datasets/__init__.py"),
        "packer": source_binding(Path(__file__)),
        "h6_dataset": source_binding(Path(h6.__file__)),
        "h6_main_pool_census": source_binding(Path(h6_census.__file__)),
        "h6_sequence_contract_v2": source_binding(Path(h4_v2.__file__)),
        "h6_sequence_contract_v1": source_binding(Path(h4_v1.__file__)),
    }


def row_identity_sha256(rows) -> str:
    """Bind row order, visible RGB leaves, and action history canonically."""
    digest = hashlib.sha256()
    for row in rows:
        payload = {
            "index": int(row.index),
            "role": str(row.role),
            "family": str(row.family),
            "scene_id": str(row.scene_id),
            "rgb": [str(row.rgb[position]) for position in POSITIONS],
            "actions": [int(row.actions[position]) for position in range(3)],
        }
        digest.update(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        )
        digest.update(b"\n")
    return digest.hexdigest()


def _require_absent(paths: list[Path]) -> None:
    occupied = [str(path) for path in paths if path.exists() or path.is_symlink()]
    if occupied:
        raise FileExistsError(
            "refusing to overwrite an existing or partial pack artifact: "
            + ", ".join(occupied)
        )


def _atomic_json(path: Path, payload: dict) -> None:
    temporary = path.with_name(path.name + ".partial")
    _require_absent([path, temporary])
    with temporary.open("x") as handle:
        handle.write(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.link(temporary, path)
    temporary.unlink()


def _publish_exclusive(temporary: Path, destination: Path) -> None:
    """Publish one completed payload without any overwrite race."""

    if temporary.is_symlink() or not temporary.is_file():
        raise FileNotFoundError(f"pack partial is absent or unsafe: {temporary}")
    os.link(temporary, destination)
    temporary.unlink()


def _read_rgb_leaf(leaf: str) -> bytes:
    """Read one canonical RGB leaf without following directory or leaf links."""
    relative = Path(leaf)
    if (
        relative.is_absolute()
        or not relative.parts
        or any(part in {"", ".", ".."} for part in relative.parts)
    ):
        raise ValueError(f"RGB leaf is not canonical relative: {leaf!r}")
    directory_flags = os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC | os.O_NOFOLLOW
    file_flags = os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW
    descriptor = os.open(RGB_ROOT, directory_flags)
    file_descriptor = None
    try:
        for component in relative.parts[:-1]:
            child = os.open(component, directory_flags, dir_fd=descriptor)
            os.close(descriptor)
            descriptor = child
        file_descriptor = os.open(relative.parts[-1], file_flags, dir_fd=descriptor)
        before = os.fstat(file_descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise ValueError(f"RGB leaf is not a regular file: {leaf}")
        chunks = []
        while True:
            chunk = os.read(file_descriptor, 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
        after = os.fstat(file_descriptor)
        if (
            (before.st_dev, before.st_ino, before.st_size)
            != (after.st_dev, after.st_ino, after.st_size)
        ):
            raise RuntimeError(f"RGB leaf changed while it was read: {leaf}")
        return b"".join(chunks)
    finally:
        if file_descriptor is not None:
            os.close(file_descriptor)
        os.close(descriptor)


def _decode_uint8(leaf: str) -> tuple[np.ndarray, dict]:
    """Decode one leaf and return its exact source-byte identity receipt."""
    from PIL import Image

    raw = _read_rgb_leaf(leaf)
    with Image.open(io.BytesIO(raw)) as image:
        if image.format != "PNG" or image.mode != "RGB" or image.size != h6.SOURCE_IMAGE_SIZE:
            raise ValueError(f"unexpected source image: {leaf}")
        image.load()
        image = image.crop(h6.CROP_BOX)
        image = image.resize(h6.MODEL_IMAGE_SIZE, Image.Resampling.BILINEAR)
        decoded = np.frombuffer(image.tobytes(), dtype=np.uint8).reshape(112, 112, 3)
    return decoded, {
        "leaf": leaf,
        "byte_count": len(raw),
        "sha256": hashlib.sha256(raw).hexdigest(),
    }


def _pack_row(job) -> tuple[int, bytes, list[dict]]:
    slot, leaves = job
    buffers, receipts = [], []
    for leaf in leaves:
        decoded, receipt = _decode_uint8(leaf)
        buffers.append(decoded.tobytes())
        receipts.append(receipt)
    return slot, b"".join(buffers), receipts


def unpack_frames(packed: np.ndarray) -> "object":
    """uint8 (..., 112, 112, 3) -> normalized float32 (..., 3, 112, 112)."""
    import torch

    tensor = torch.from_numpy(np.ascontiguousarray(packed))
    tensor = tensor.permute(*range(tensor.ndim - 3), tensor.ndim - 1,
                            tensor.ndim - 3, tensor.ndim - 2)
    tensor = tensor.contiguous().to(dtype=torch.float32).div_(255.0)
    mean = torch.tensor(IMAGENET_MEAN, dtype=torch.float32).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD, dtype=torch.float32).view(3, 1, 1)
    return tensor.sub_(mean).div_(std)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--verify", type=int, default=24,
                        help="rows to bit-compare against the frozen decoder")
    parser.add_argument("--out", default=str(OUTPUT_ROOT))
    args = parser.parse_args()

    if args.workers < 1:
        raise ValueError("--workers must be at least one")
    if args.verify < 1:
        raise ValueError("--verify must be at least one for a provenance-bound pack")

    output_root = Path(args.out).resolve()
    try:
        output_root.relative_to((REPO_ROOT / ".generated/dev").resolve())
    except ValueError as exc:
        raise ValueError("--out must stay under .generated/dev") from exc
    output_root.mkdir(parents=True, exist_ok=False)
    manifest_path = output_root / "manifest.json"
    _require_absent([manifest_path, manifest_path.with_name("manifest.json.partial")])
    sources = pack_source_bindings()
    manifest = {}
    for role in ("train", "val"):
        rows, index_audit = h6.load_bound_index(REPO_ROOT, role=role)
        count = len(rows)
        print(f"{role}: {count} rows x {len(POSITIONS)} positions", flush=True)

        frames_path = output_root / f"{role}_frames.u8"
        frames_partial = frames_path.with_name(frames_path.name + ".partial")
        actions_path = output_root / f"{role}_actions.npy"
        actions_partial = actions_path.with_name(actions_path.name + ".partial")
        meta_path = output_root / f"{role}_meta.json"
        meta_partial = meta_path.with_name(meta_path.name + ".partial")
        _require_absent([
            frames_path, frames_partial, actions_path, actions_partial,
            meta_path, meta_partial,
        ])
        frame_shape = (count, len(POSITIONS), 112, 112, 3)
        expected_frame_bytes = count * len(POSITIONS) * FRAME_BYTES
        descriptor = os.open(
            frames_partial,
            os.O_RDWR | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0),
            0o600,
        )
        try:
            os.ftruncate(descriptor, expected_frame_bytes)
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        frames = np.memmap(
            frames_partial, dtype=np.uint8, mode="r+", shape=frame_shape
        )
        actions = np.zeros((count, 3), dtype=np.int64)
        scene_ids, families = [], []
        jobs = []
        for slot, row in enumerate(rows):
            jobs.append((slot, [row.rgb[p] for p in POSITIONS]))
            actions[slot] = [row.actions[i] for i in range(3)]
            scene_ids.append(row.scene_id)
            families.append(row.family)

        done = 0
        source_rgb_identity = hashlib.sha256()
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            for slot, blob, receipts in pool.map(_pack_row, jobs, chunksize=16):
                frames[slot] = np.frombuffer(blob, dtype=np.uint8).reshape(
                    len(POSITIONS), 112, 112, 3)
                for position, receipt in zip(POSITIONS, receipts, strict=True):
                    source_rgb_identity.update(json.dumps(
                        {"slot": slot, "position": position, **receipt},
                        sort_keys=True,
                        separators=(",", ":"),
                    ).encode("utf-8"))
                    source_rgb_identity.update(b"\n")
                done += 1
                if done % 2000 == 0:
                    print(f"  {done}/{count}", flush=True)
        frames.flush()

        worst = None
        if args.verify:
            import torch
            rng = np.random.default_rng(20260731)
            picks = rng.choice(count, size=min(args.verify, count), replace=False)
            worst = 0.0
            for slot in picks:
                row = rows[int(slot)]
                for pi, p in enumerate(POSITIONS):
                    reference = h6.rectify_h6_rgb_bytes(
                        _read_rgb_leaf(row.rgb[p]))
                    got = unpack_frames(np.asarray(frames[slot, pi]))
                    worst = max(worst, float((reference - got).abs().max()))
            print(f"  verify {len(picks)} rows: max abs deviation {worst:.3e}",
                  flush=True)
            if worst > 0.0:
                raise SystemExit("pack is not bit-exact against frozen decoder")

        frames.flush()
        del frames
        with actions_partial.open("xb") as handle:
            np.save(handle, actions, allow_pickle=False)
        with meta_partial.open("x") as handle:
            handle.write(json.dumps(
                {"scene_ids": scene_ids, "families": families},
                sort_keys=True, separators=(",", ":")) + "\n")
        frames_bytes = frames_partial.stat().st_size
        if frames_bytes != expected_frame_bytes:
            raise RuntimeError(
                f"{role} frame pack byte count changed: {frames_bytes} != "
                f"{expected_frame_bytes}"
            )
        role_manifest = {
            "rows": count,
            "positions": list(POSITIONS),
            "row_identity_sha256": row_identity_sha256(rows),
            "source_rgb": {
                "leaf_count": count * len(POSITIONS),
                "ordered_identity_sha256": source_rgb_identity.hexdigest(),
            },
            "index_binding": {
                "role": index_audit["role"],
                "path": index_audit["path"],
                "byte_count": int(index_audit["byte_count"]),
                "file_sha256": index_audit["file_sha256"],
            },
            "frames": {
                "path": frames_path.name,
                "byte_count": frames_bytes,
                "sha256": sha256_file(frames_partial),
                "dtype": "uint8",
                "shape": [count, len(POSITIONS), 112, 112, 3],
            },
            "actions": {
                "path": actions_path.name,
                "byte_count": actions_partial.stat().st_size,
                "sha256": sha256_file(actions_partial),
                "dtype": "int64",
                "shape": [count, 3],
            },
            "metadata": {
                "path": meta_path.name,
                "byte_count": meta_partial.stat().st_size,
                "sha256": sha256_file(meta_partial),
            },
            "verification": {
                "seed": 20260731,
                "requested_rows": int(args.verify),
                "sampled_rows": min(max(int(args.verify), 0), count),
                "max_abs_deviation": worst,
            },
        }
        _publish_exclusive(frames_partial, frames_path)
        _publish_exclusive(actions_partial, actions_path)
        _publish_exclusive(meta_partial, meta_path)
        manifest[role] = role_manifest

    current_sources = pack_source_bindings()
    if current_sources != sources:
        raise RuntimeError("packer source changed while the pack was generated")
    _atomic_json(manifest_path, {
        "schema": PACK_SCHEMA,
        "citable_as_scientific_evidence": False,
        "source_layout": {
            "rgb_root": RGB_ROOT.relative_to(REPO_ROOT).as_posix(),
            "visible_positions": list(POSITIONS),
            "forbidden_positions_opened": [],
        },
        "runtime": {
            "numpy_version": np.__version__,
            "pillow_version": PIL.__version__,
        },
        "sources": sources,
        "roles": manifest,
    })
    print(f"wrote {output_root}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
