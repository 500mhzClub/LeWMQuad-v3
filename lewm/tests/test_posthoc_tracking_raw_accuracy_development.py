"""Synthetic checks for the distinct post-hoc adapter; no retained data reads."""
import ast
from copy import deepcopy
import inspect

import numpy as np
import pytest

from lewm import independent_tracking_numerical_verification_development as numerical
from lewm.posthoc_tracking_numerical_view_development import sensor_convention_view
from lewm.tests.test_independent_tracking_numerical_verification_development import pose_case
from scripts import independent_tracking_evaluation_development as original
from scripts import posthoc_tracking_raw_audit_development as adapted
from scripts import read_go2_tracking_posthoc_raw_accuracy_v1 as reader


def test_raw_audit_has_only_declared_function_docstring_and_coverage_change():
    old = ast.parse(inspect.getsource(original._raw_audit)).body[0]
    new = ast.parse(inspect.getsource(adapted.raw_audit_sensor_convention)).body[0]
    old.name = new.name
    old.body[0] = new.body[0]
    changed = 0
    for node in ast.walk(old):
        if isinstance(node, ast.Name) and node.id == 'measured_coverage':
            node.id = 'coverage_with_sensor_rotation_convention'
            changed += 1
    assert changed == 1
    assert ast.dump(old) == ast.dump(new)


def test_native_roundoff_still_fails_old_gate_but_private_view_checks_real_3d_scores(pose_case):
    raw, rows, evaluated, phase, score = pose_case
    raw['base_pose_world'][:, 3:] *= 1 + 1.15e-7
    before = raw['base_pose_world'].copy()
    with pytest.raises(ValueError, match='unit native'):
        numerical.verify_pose_stream(raw, rows, evaluated, phase, score)
    view, provenance = sensor_convention_view(raw)
    result = numerical.verify_pose_stream(view, rows, evaluated, phase, score)
    assert result['numerical_pose_score_reconstruction_verified']
    assert not provenance['original_frozen_norm_gate_passes']
    assert not provenance['independent_native_data']
    assert not provenance['original_attempt_passed']
    assert not view['base_pose_world'].flags.writeable
    assert not np.shares_memory(view['base_pose_world'], raw['base_pose_world'])
    np.testing.assert_array_equal(raw['base_pose_world'], before)
    assert result['score']['arms']['original']['position_m']['count'] == 8
    assert result['score']['arms']['original']['paired_position_m']['count'] == 8
    assert not result['score']['empirical_local_pose_allocation_met']['original']


@pytest.mark.parametrize('bad', [0., .5, 1.01, float('nan'), float('inf')])
def test_private_view_rejects_invalid_native_orientation(bad):
    pose = np.zeros((2, 7)); pose[:, 6] = bad
    with pytest.raises(ValueError):
        sensor_convention_view({'base_pose_world': pose})


@pytest.mark.parametrize('fault', ['clock', 'score', 'row_missing', 'availability', 'resurrection'])
def test_representation_interface_preserves_negative_numerical_gates(pose_case, fault):
    raw, rows, evaluated, phase, score = deepcopy(pose_case)
    raw['base_pose_world'][:, 3:] *= 1 + 1.15e-7
    view, _ = sensor_convention_view(raw)
    if fault == 'clock': view['timestamp_s'][10] += .001
    elif fault == 'score': score['arms']['temporal_anchor']['position_m']['mean'] += .001
    elif fault == 'row_missing': evaluated.pop()
    elif fault == 'availability': phase['arms']['original']['available'] = 12
    else:
        rows[10]['arms']['original']['pose'] = rows[10]['arms']['temporal_anchor']['pose']
        rows[10]['availability'] = 'both'
    with pytest.raises(ValueError):
        numerical.verify_pose_stream(view, rows, evaluated, phase, score)


def test_admission_source_failure_stops_before_original_artifact_or_native_access(monkeypatch):
    called = []
    def reject(bindings):
        called.append('source')
        raise ValueError('source identity changed')
    monkeypatch.setattr(reader, 'source_check', reject)
    monkeypatch.setattr(reader.admission, 'admitted_inputs', lambda: pytest.fail('artifact access before source admission'))
    with pytest.raises(ValueError, match='source identity'):
        reader.admit()
    assert called == ['source']


def test_new_destination_excludes_old_root_and_rejects_duplicate_writes(tmp_path, monkeypatch):
    monkeypatch.setattr(reader.custody, 'BASE', tmp_path)
    target = tmp_path / 'go2_synthetic_attempt_001'; target.mkdir()
    old = tmp_path / 'go2_original_attempt_001'; old.mkdir()
    monkeypatch.setattr(reader, 'INPUT', old)
    with pytest.raises(ValueError, match='distinct derived'):
        reader.ScoreDestination(old, reader.base.TRIALS[0])
    store = reader.ScoreDestination(target, reader.base.TRIALS[0])
    store.scenario = 'base'
    with store.stream(store.trial + '_evaluation.jsonl') as stream:
        store.append(stream, {'failed_arm': None})
    assert store.output == old and (target / store.name()).is_file()
    assert list(old.iterdir()) == []  # Explicit fresh synthetic root only.
    with pytest.raises(FileExistsError):
        with store.stream(store.trial + '_evaluation.jsonl'):
            pass


def test_worker_raw_failure_retains_whole_unscored_population(tmp_path, monkeypatch):
    monkeypatch.setattr(reader.custody, 'BASE', tmp_path)
    target = tmp_path / 'go2_synthetic_attempt_001'; target.mkdir()
    monkeypatch.setattr(reader, 'source_check', lambda sources: None)
    def reject(*args):
        raise ValueError('synthetic raw contact discrepancy')
    monkeypatch.setattr(reader, 'raw_audit_sensor_convention', reject)
    monkeypatch.setattr(reader.scoring, '_score_trial', lambda *args: pytest.fail('scoring after raw failure'))
    report = reader.analyze_tape(target, reader.base.TRIALS[0], {}, {}, 'a'*64, {})
    assert report['status'] == 'POSTHOC_TAPE_ANALYSIS_FAILED'
    assert report['stage'] == 'raw_sensor_contact_geometry_setup_command_stop_raster_audit'
    assert report['streams'] == {}
    assert report['uncompleted_scenarios'] == list(reader.SCENARIOS)
    assert len(report['uncompleted_scenarios']) == 12


def test_source_check_rejects_protected_names_before_open():
    with pytest.raises(ValueError, match='nonprotected'):
        reader.source_check({'sealed_x/no.py': '0'*64})
