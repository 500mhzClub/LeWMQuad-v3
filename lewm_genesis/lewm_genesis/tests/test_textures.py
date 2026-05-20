"""Tests for render-time surface texture selection + UV cube geometry."""

from __future__ import annotations

from pathlib import Path

from lewm_genesis import textures as T


def test_category_for_kind_only_textures_structural_surfaces():
    assert T.category_for_kind("wall") == "wall"
    assert T.category_for_kind("box_obstacle") == "obstacle"
    assert T.category_for_kind("ramp") == "obstacle"
    assert T.category_for_kind("step") == "obstacle"
    # Identity/decoy surfaces stay solid.
    for solid in ("landmark", "distractor", "slick_patch", "", "unknown"):
        assert T.category_for_kind(solid) is None


def _fake_texture_root(tmp_path: Path) -> Path:
    for cat in T.CATEGORIES:
        d = tmp_path / cat
        d.mkdir(parents=True)
        for name in ("a", "b", "c"):
            (d / f"{name}.jpg").write_bytes(b"\xff\xd8\xff")  # minimal JPEG magic
    return tmp_path


def test_selection_is_deterministic_and_seed_varying(tmp_path, monkeypatch):
    root = _fake_texture_root(tmp_path)
    monkeypatch.setenv("LEWM_TEXTURE_ROOT", str(root))
    T.available_textures.cache_clear()

    s1 = T.select_scene_textures(visual_seed=7, scene_id="scene-x")
    s2 = T.select_scene_textures(visual_seed=7, scene_id="scene-x")
    s3 = T.select_scene_textures(visual_seed=8, scene_id="scene-x")
    s4 = T.select_scene_textures(visual_seed=7, scene_id="scene-y")

    assert s1 == s2  # reproducible
    assert set(s1) == set(T.CATEGORIES)
    assert all(v is not None for v in s1.values())
    assert s1 != s3 or s1 != s4  # varies with seed and/or scene id


def test_selection_handles_missing_assets(tmp_path, monkeypatch):
    monkeypatch.setenv("LEWM_TEXTURE_ROOT", str(tmp_path / "does_not_exist"))
    T.available_textures.cache_clear()
    sel = T.select_scene_textures(visual_seed=1, scene_id="s")
    assert sel == {cat: None for cat in T.CATEGORIES}


def test_box_obj_text_has_six_quads_with_uvs():
    obj = T.box_obj_text((0.1, 2.4, 1.0), tiles_per_m=0.7)
    verts = [ln for ln in obj.splitlines() if ln.startswith("v ")]
    uvs = [ln for ln in obj.splitlines() if ln.startswith("vt ")]
    faces = [ln for ln in obj.splitlines() if ln.startswith("f ")]
    assert len(verts) == 24  # 6 faces * 4 corners
    assert len(uvs) == 24
    assert len(faces) == 12  # 6 faces * 2 triangles
    # UVs scale with face size * tiles_per_m: the 2.4 m edge tiles ~1.68x.
    max_u = max(float(ln.split()[1]) for ln in uvs)
    assert max_u > 1.0


def test_cached_box_obj_is_written_and_stable(tmp_path):
    p1 = T.cached_box_obj((0.1, 2.4, 1.0), cache_dir=tmp_path)
    p2 = T.cached_box_obj((0.1, 2.4, 1.0), cache_dir=tmp_path)
    assert p1 == p2
    assert Path(p1).is_file()
    assert Path(p1).read_text().count("\nf ") == 12
