from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import PIL
import pytest
import torch

from scripts import dev_pack_h6_temporal_frames as packer
from scripts import dev_train_temporal_jepa_scaled as trainer


def _rows(role: str) -> tuple[SimpleNamespace, ...]:
    return tuple(
        SimpleNamespace(
            index=index,
            role=role,
            family=f"family_{index}",
            scene_id=f"{role}_scene_{index}",
            rgb=tuple(
                f"{role}_scene_{index}/rgb/frame_{position:03d}.png"
                for position in range(7)
            ),
            actions=(index, index + 1, index + 2, 3, 4, 5),
        )
        for index in range(2)
    )


def _artifact(path: Path, **schema) -> dict:
    return {
        "path": path.name,
        "byte_count": path.stat().st_size,
        "sha256": packer.sha256_file(path),
        **schema,
    }


def _write_synthetic_pack(
    root: Path,
) -> tuple[dict[str, SimpleNamespace], dict[str, tuple[SimpleNamespace, ...]]]:
    role_rows = {role: _rows(role) for role in ("train", "val")}
    bindings = {
        role: SimpleNamespace(
            role=role,
            path=Path(f"synthetic/{role}.jsonl"),
            byte_count=100 + index,
            sha256=(str(index + 1) * 64),
            row_count=len(role_rows[role]),
        )
        for index, role in enumerate(("train", "val"))
    }
    roles = {}
    for role, rows in role_rows.items():
        frames_path = root / f"{role}_frames.u8"
        actions_path = root / f"{role}_actions.npy"
        metadata_path = root / f"{role}_meta.json"
        frames_path.write_bytes(
            bytes(len(rows) * len(packer.POSITIONS) * packer.FRAME_BYTES)
        )
        actions = np.asarray(
            [[row.actions[position] for position in range(3)] for row in rows],
            dtype=np.int64,
        )
        with actions_path.open("wb") as handle:
            np.save(handle, actions, allow_pickle=False)
        metadata_path.write_text(
            json.dumps(
                {
                    "scene_ids": [row.scene_id for row in rows],
                    "families": [row.family for row in rows],
                },
                sort_keys=True,
                separators=(",", ":"),
            )
            + "\n"
        )
        binding = bindings[role]
        roles[role] = {
            "rows": len(rows),
            "positions": list(packer.POSITIONS),
            "row_identity_sha256": packer.row_identity_sha256(rows),
            "source_rgb": {
                "leaf_count": len(rows) * len(packer.POSITIONS),
                "ordered_identity_sha256": ("a" if role == "train" else "b")
                * 64,
            },
            "index_binding": {
                "role": binding.role,
                "path": binding.path.as_posix(),
                "byte_count": binding.byte_count,
                "file_sha256": binding.sha256,
            },
            "frames": _artifact(
                frames_path,
                dtype="uint8",
                shape=[len(rows), len(packer.POSITIONS), 112, 112, 3],
            ),
            "actions": _artifact(
                actions_path,
                dtype="int64",
                shape=[len(rows), 3],
            ),
            "metadata": _artifact(metadata_path),
            "verification": {
                "seed": 20260731,
                "requested_rows": 2,
                "sampled_rows": 2,
                "max_abs_deviation": 0.0,
            },
        }

    manifest = {
        "schema": packer.PACK_SCHEMA,
        "citable_as_scientific_evidence": False,
        "source_layout": {
            "rgb_root": packer.RGB_ROOT.relative_to(trainer.REPO_ROOT).as_posix(),
            "visible_positions": list(packer.POSITIONS),
            "forbidden_positions_opened": [],
        },
        "runtime": {
            "numpy_version": np.__version__,
            "pillow_version": PIL.__version__,
        },
        "sources": {
            **packer.pack_source_bindings(),
        },
        "roles": roles,
    }
    (root / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    return bindings, role_rows


def test_pack_helpers_bind_order_actions_and_regular_files(tmp_path: Path) -> None:
    rows = _rows("train")
    digest = packer.row_identity_sha256(rows)
    assert digest == packer.row_identity_sha256(rows)
    assert digest != packer.row_identity_sha256(tuple(reversed(rows)))

    changed = list(rows)
    changed[0] = SimpleNamespace(**vars(rows[0]))
    changed[0].actions = (8, *changed[0].actions[1:])
    assert digest != packer.row_identity_sha256(changed)

    regular = tmp_path / "artifact.bin"
    regular.write_bytes(b"bound bytes")
    symlink = tmp_path / "artifact-link.bin"
    symlink.symlink_to(regular)
    assert len(packer.sha256_file(regular)) == 64
    with pytest.raises(ValueError, match="non-symlink"):
        packer.sha256_file(symlink)
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        packer._require_absent([regular])


def test_rgb_leaf_reader_refuses_leaf_and_parent_symlinks(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rgb_root = tmp_path / "rgb-root"
    scene = rgb_root / "scene" / "rgb"
    scene.mkdir(parents=True)
    leaf = scene / "frame.png"
    leaf.write_bytes(b"synthetic png bytes")
    monkeypatch.setattr(packer, "RGB_ROOT", rgb_root)

    assert packer._read_rgb_leaf("scene/rgb/frame.png") == b"synthetic png bytes"

    linked_leaf = scene / "linked.png"
    linked_leaf.symlink_to(leaf)
    with pytest.raises(OSError):
        packer._read_rgb_leaf("scene/rgb/linked.png")

    linked_scene = rgb_root / "linked-scene"
    linked_scene.symlink_to(rgb_root / "scene", target_is_directory=True)
    with pytest.raises(OSError):
        packer._read_rgb_leaf("linked-scene/rgb/frame.png")


def test_v2_pack_validation_and_loading_are_hash_bound(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bindings, role_rows = _write_synthetic_pack(tmp_path)

    def load_bound_index(_repo_root: Path, *, role: str):
        binding = bindings[role]
        return role_rows[role], {
            "role": role,
            "path": binding.path.as_posix(),
            "byte_count": binding.byte_count,
            "file_sha256": binding.sha256,
            "row_count": binding.row_count,
        }

    monkeypatch.setattr(trainer.h6, "INDEX_BINDINGS", bindings)
    monkeypatch.setattr(trainer.h6, "load_bound_index", load_bound_index)

    validated = trainer.validate_pack_role(tmp_path, "train")
    assert validated["role"]["row_identity_sha256"] == packer.row_identity_sha256(
        role_rows["train"]
    )
    frames, actions, provenance = trainer.load_pack(
        tmp_path, "train", torch.device("cpu")
    )
    assert tuple(frames.shape) == (2, 4, 112, 112, 3)
    assert actions.tolist() == [[0, 1, 2], [1, 2, 3]]
    assert provenance["index_binding"]["role"] == "train"

    frames_path = tmp_path / "train_frames.u8"
    with frames_path.open("r+b") as handle:
        handle.seek(-1, 2)
        handle.write(b"\x01")
    with pytest.raises(ValueError, match="bytes or identity changed"):
        trainer.validate_pack_role(tmp_path, "train")
