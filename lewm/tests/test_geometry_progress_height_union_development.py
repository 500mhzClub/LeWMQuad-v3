"""Constructor successor preserves the failed run's scientific/audit contract."""
import ast
from pathlib import Path
import pytest
from scripts import run_go2_geometry_progress_height_union_v1 as run
from scripts import run_go2_geometry_progress_near_field_v1 as old
from scripts import audit_go2_geometry_progress_height_union_v1 as audit
from scripts import audit_go2_geometry_progress_near_field_v1 as old_audit
from scripts.navigation_artifact_root_development import BASE


def function(source,name):
    return ast.dump(next(n for n in ast.parse(source).body if isinstance(n,ast.FunctionDef) and n.name==name))


def test_collection_logic_and_artifact_roster_unchanged_except_explicit_session():
    a=Path(run.__file__).read_text().replace('GeometryProgressHeightUnionSession','GeometryProgressNearFieldSession')
    a=a.replace('GEOMETRY_PROGRESS_HEIGHT_UNION','GEOMETRY_PROGRESS_NEAR_FIELD')
    b=Path(old.__file__).read_text()
    for name in ('collect','artifacts'):assert function(a,name)==function(b,name)
    assert run.specification is old.specification and run.decision is old.decision and run.schedule is old.schedule
    assert run.OUTPUT==BASE/'go2_geometry_progress_height_union_v1_attempt_001' and run.OUTPUT!=old.OUTPUT


def test_all_raw_and_raster_condition_checks_reused_unchanged():
    a=Path(audit.__file__).read_text();b=Path(old_audit.__file__).read_text()
    for name in ('audit_condition','validate_rasters','audit_rasters_and_footprints'):
        assert function(a,name)==function(b,name)
    assert audit.audit_commands is old_audit.audit_commands and audit.measurement_gate is old_audit.measurement_gate
    assert audit.INPUT==run.OUTPUT


def test_new_native_constructor_uses_only_the_reviewed_variable_height_visual_provider():
    a=Path('scripts/geometry_progress_height_union_session_development.py').read_text()
    b=Path('scripts/geometry_progress_near_field_session_development.py').read_text()
    a=a.replace('GeometryProgressHeightUnion','GeometryProgressNearField').replace('variable_height_union_rgbd_scene_development','union_wall_rgbd_scene_development')
    assert ast.dump(ast.parse(a))==ast.dump(ast.parse(b))
    assert run.BENCH_IDS['result.json']=='8cfb8b30f67514a4f64490eea32eb3f188e57a32f57894d29dc29f0b20231738'


def test_refuses_existing_attempt_before_preflight(monkeypatch,tmp_path):
    monkeypatch.setattr(run,'OUTPUT',tmp_path)
    monkeypatch.setattr(run,'preflight',lambda:pytest.fail('existing attempt cannot reach preflight'))
    with pytest.raises(ValueError,match='exclusive'):run.main()
