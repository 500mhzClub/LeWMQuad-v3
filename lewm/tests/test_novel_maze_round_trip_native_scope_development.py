import ast
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest
from lewm.novel_maze_round_trip_contract_development import MAX_OBSERVATIONS, MAX_COMMAND_TICKS, NAVIGATION_TICKS
from scripts import novel_maze_round_trip_session_development as session
from scripts.novel_maze_round_trip_command_audit_development import audit_commands

ROOT = Path(__file__).resolve().parents[2]


def test_narrow_physical_source_changes_only_scene_definition_and_labels():
    original = (ROOT/'scripts/auxiliary_depth_visible_robot_session_development.py').read_text()
    expected = original.replace('from lewm.geometry_progress_layout_family_development import specification,pack',
        'from lewm.novel_maze_round_trip_scene_development import specification,pack')
    expected = expected.replace('VisibleRobotFamilyPhysicalInit', 'NovelMazeRoundTripPhysicalInit')
    expected = expected.replace('VisibleRobotFamilySession', 'NovelMazeBaseSession')
    expected = expected.replace("specification(spec['trial'])", "specification(spec['layout_index'])")
    expected = expected.replace('exact CPU geometry-progress experiment required', 'exact CPU prospective maze experiment required')
    expected = expected.replace('fixed recorded command prefix for sensor characterization; no navigation policy',
        'observed round-trip maze controller; evaluator-only native guards')
    assert ast.dump(ast.parse(expected)) == ast.dump(ast.parse(
        (ROOT/'scripts/novel_maze_round_trip_physical_session_development.py').read_text()))


def test_auxiliary_packet_and_command_audit_keep_original_checks():
    def function(path, name):
        return next(n for n in ast.parse((ROOT/path).read_text()).body if isinstance(n, ast.FunctionDef) and n.name == name)
    old = function('scripts/auxiliary_downward45_packet_replay_development.py', 'packet')
    new = function('scripts/novel_maze_auxiliary_packet_development.py', 'packet')
    # Only the explicit bounded population expression changes.
    old.body[0].test = new.body[0].test
    assert ast.dump(old) == ast.dump(new)
    old = function('scripts/overlap_retention_goal_audit_development.py', 'audit_commands')
    for node in ast.walk(old):
        if isinstance(node, ast.Constant) and node.value == 'online_learned_goal_command':
            node.value = 'online_learned_round_trip_command'
    assert ast.dump(old) == ast.dump(function('scripts/novel_maze_round_trip_command_audit_development.py', 'audit_commands'))


@pytest.mark.parametrize('index', [253, 254, MAX_OBSERVATIONS-1])
def test_paired_acquisition_crosses_old_bound_with_exact_native_clock(monkeypatch, index):
    c = object.__new__(session.NovelMazeRoundTripSession)
    c.samples = [None]*(750+50*index); c.model_manifest = [None]*(index+1)
    c.auxiliary_audit = [None]*index; c.output = Path('/synthetic-only')
    now = 1_500_000_000+index*100_000_000
    monkeypatch.setattr(session.NovelMazeBaseSession, 'sensor_packets', lambda self: ({}, {}, {}, now))
    monkeypatch.setattr(session, 'capture', lambda *a: dict(frame=index, measured_ns=now,
        physical_sample_index=749+50*index, native_depth_sha256='a'*64, calibration_id='synthetic'))
    monkeypatch.setattr(session, 'packet', lambda directory, i, policy, acquisition, **kw:
        dict(frame=i, clock=kw['now_ns']))
    assert c.sensor_packets()[3] == dict(frame=index, clock=now)
    assert len(c.auxiliary_audit) == index+1


def test_acquisition_over_bound_stops_before_any_render(monkeypatch):
    c = object.__new__(session.NovelMazeRoundTripSession)
    c.samples = [None]*(750+50*MAX_OBSERVATIONS)
    monkeypatch.setattr(session.NovelMazeBaseSession, 'sensor_packets', lambda self: pytest.fail('over-bound acquisition'))
    with pytest.raises(ValueError): c.sensor_packets()


def test_long_complete_command_audit_and_tampered_dispatch():
    commands = MAX_COMMAND_TICKS; n = 750+50*commands
    raw = dict(timestamp_s=np.arange(1, n+1)*.002, phase=np.zeros(n, int))
    for k in ('requested_command', 'applied_command', 'post_slew_applied_command'):
        raw[k] = np.zeros((n, 3), np.float64)
    tape = []; rows = []
    for i in range(commands+1):
        terminal = 'MISSION_TICK_BUDGET_EXHAUSTED' if i >= 3+NAVIGATION_TICKS else None
        rows.append(dict(tick=i, decision=dict(terminal=terminal, requested_command=[0., 0., 0.])))
        if i == commands: break
        phase, role = (3, 'terminal_zero_drain') if terminal else ((1, 'causal_history_warmup') if i < 3
            else (2, 'online_learned_round_trip_command'))
        start = 749+50*i
        raw['phase'][start+1:start+51] = phase
        tape.append(dict(tick=i, requested_command=[0., 0., 0.], phase=phase, role=role,
            pre_sample_index=start, post_sample_index=start+50, completed=True))
    result = dict(command_ticks=commands, decisions=commands+1, completed_ticks=commands, terminal_zero_ticks=10,
        physical_stop=None, acquisition_stop=None, schedule_terminal='MISSION_TICK_BUDGET_EXHAUSTED')
    audit_commands(raw, tape, rows, result)
    raw['post_slew_applied_command'][750+50*254, 0] = .1
    with pytest.raises(AssertionError): audit_commands(raw, tape, rows, result)
