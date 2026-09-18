"""Compare actual executions up to the original deadline; never infer return."""
from contextlib import closing
from copy import deepcopy
from itertools import islice

import numpy as np

from scripts import extended_budget_anchored_maze_development as extended
from scripts.reached_frontier_native_prefix_development import public_packets as original_public_packets
from scripts.maze_decision_stream_development import read_rows as original_rows
from scripts.startup_raw_sensor_audit_development import read_json
from scripts.navigation_artifact_root_development import artifact_path, verify_artifacts
from scripts.audit_go2_independent_pulse_context_pilot_v1 import fingerprint

ORIGINAL_BUDGET = 3000
BOUNDARY = 3+ORIGINAL_BUDGET
FRAMES = BOUNDARY+1
PHYSICS_SAMPLES = 750+50*BOUNDARY
BUDGET_PATHS = ('shared_navigation_budget_ticks', 'mission_receipt.global_navigation_ticks')
public_packets = extended._bind(original_public_packets,
    IntentReturnRGBDReplay=extended.ExtendedBudgetRGBDReplay, packet=extended.rgb_packet)


def normalize(decision, *, budget):
    """Exactly two declared scalar paths; unrelated budget fields stay intact."""
    if type(budget) is not int or budget not in (ORIGINAL_BUDGET, extended.NAVIGATION_TICKS):
        raise ValueError('exact original or extended budget required')
    value = deepcopy(decision)
    if (type(value) is not dict or type(value.get('mission_receipt')) is not dict
            or type(value.get('shared_navigation_budget_ticks')) is not int
            or value['shared_navigation_budget_ticks'] != budget
            or type(value['mission_receipt'].get('global_navigation_ticks')) is not int
            or value['mission_receipt']['global_navigation_ticks'] != budget):
        raise ValueError('both exact declared mission budget fields required')
    value['shared_navigation_budget_ticks'] = ORIGINAL_BUDGET
    value['mission_receipt']['global_navigation_ticks'] = ORIGINAL_BUDGET
    return value


class BudgetPrefixComparison:
    """Record divergence without relabeling it as a successful budget control."""
    def __init__(self):
        self.frames = 0
        self.first_decision_difference = None
        self.first_request_or_terminal_difference = None
        self.forecasts = 0
        self.boundary = None

    def observe(self, old, new):
        frame = self.frames
        if frame >= FRAMES:
            raise ValueError('no observation after the original intervention boundary')
        for row in (old, new):
            if (type(row.get('tick')) is not int or row['tick'] != frame
                    or type(row.get('observation_index')) is not int or row['observation_index'] != frame
                    or type(row.get('pre_sample_index')) is not int or row['pre_sample_index'] != 749+50*frame
                    or type(row.get('decision')) is not dict
                    or type(row['decision'].get('tick')) is not int or row['decision']['tick'] != frame):
                raise ValueError('complete consecutive original observation identities required')
        a, b = normalize(old['decision'], budget=ORIGINAL_BUDGET), normalize(new['decision'], budget=extended.NAVIGATION_TICKS)
        same = fingerprint(a) == fingerprint(b)
        command_same = fingerprint((a['requested_command'], a['terminal'])) == fingerprint((b['requested_command'], b['terminal']))
        if not same and self.first_decision_difference is None: self.first_decision_difference = frame
        if not command_same and self.first_request_or_terminal_difference is None:
            self.first_request_or_terminal_difference = frame
        if frame < BOUNDARY:
            if a['terminal'] is not None:
                raise ValueError('original must remain nonterminal before its recorded budget boundary')
            if a.get('new_selection') is not None and same: self.forecasts += 1
        else:
            if a['terminal'] != 'MISSION_TICK_BUDGET_EXHAUSTED' or a['requested_command'] != [0., 0., 0.]:
                raise ValueError('original exhausted-budget zero-command boundary required')
            self.boundary = dict(frame=frame, original_terminal=a['terminal'], candidate_terminal=b['terminal'],
                original_requested_command=a['requested_command'], candidate_requested_command=b['requested_command'],
                candidate_continues=b['terminal'] is None)
        self.frames += 1
        return dict(frame=frame, complete_normalized_decision_exact=same, requested_command_and_terminal_exact=command_same)

    def report(self):
        if self.frames != FRAMES or self.boundary is None:
            raise ValueError('complete through-boundary paired observations required')
        exact = self.first_decision_difference in (None, BOUNDARY)
        return dict(frames=self.frames, normalized_budget_paths=list(BUDGET_PATHS),
            all_preboundary_decisions_exact=exact,
            first_normalized_decision_difference=self.first_decision_difference,
            first_requested_command_or_terminal_difference=self.first_request_or_terminal_difference,
            equal_preboundary_forecast_decisions=self.forecasts, boundary=deepcopy(self.boundary),
            following_observations_compared=False, unexecuted_outcomes_inferred=False,
            verified_round_trip=False, learned_planning_or_memory_advantage_established=False)


def command_prefix(tapes):
    if len(tapes) != 2 or any(len(t) < BOUNDARY for t in tapes):
        raise ValueError('both complete command populations before original cutoff required')
    first = None
    for i in range(BOUNDARY):
        for tape in tapes:
            t = tape[i]
            if (type(t['tick']) is not int or t['tick'] != i or t['completed'] is not True
                    or type(t['pre_sample_index']) is not int or t['pre_sample_index'] != 749+50*i
                    or type(t['post_sample_index']) is not int or t['post_sample_index'] != 799+50*i):
                raise ValueError('actual completed preboundary command endpoints required')
        if fingerprint(tapes[0][i]) != fingerprint(tapes[1][i]) and first is None: first = i
    return dict(preboundary_commands=BOUNDARY, first_complete_command_difference=first,
        all_preboundary_commands_exact=first is None)


def compare(prior, current, *, prior_bindings, current_bindings):
    """Caller supplies audited episode bindings and authenticates launch/models.

All supplied files are rehashed before and after. This compares already
executed prefixes and reads no observation following the original cutoff.
"""
    directories = (prior, current); bindings = (prior_bindings, current_bindings)
    required = {'physics_trace.npz', 'command_tape.json', 'context_decisions.jsonl.gz',
        'policy_observations.json', 'policy_histories.npz', 'depth_observations.json',
        'fast_gyro_histories.npz', 'auxiliary_camera_audit.json'}
    required |= {f'{kind}_{i:04d}.{suffix}' for i in range(FRAMES)
        for kind, suffix in [('rgb', 'png'), ('depth', 'npz'), ('auxiliary_rgb', 'png'), ('auxiliary_depth', 'npz')]}
    for directory, ids in zip(directories, bindings, strict=True):
        if not {directory.name+'/'+name for name in required} <= ids.keys():
            raise ValueError('every consumed raw prefix input must have an explicit binding')
        verify_artifacts(directory.parent, ids)
    physics = []
    for directory in directories:
        with np.load(artifact_path(directory.parent, directory.name+'/physics_trace.npz'), allow_pickle=False) as raw:
            if not raw.files or any(not PHYSICS_SAMPLES <= len(raw[k]) <= 201400 for k in raw.files):
                raise ValueError('bounded complete shared physics population required')
            physics.append(fingerprint({k:raw[k][:PHYSICS_SAMPLES] for k in raw.files}))
    tapes = [read_json(p, 'command_tape.json') for p in directories]
    commands = command_prefix(tapes); comparator = BudgetPrefixComparison(); first_public = None
    with closing(original_rows(prior)) as old_rows, closing(extended.read_rows(current)) as new_rows:
        for i, (old, new, a, b) in enumerate(zip(islice(old_rows, FRAMES), islice(new_rows, FRAMES),
                public_packets(prior, FRAMES), public_packets(current, FRAMES), strict=True)):
            comparator.observe(old, new)
            if a != b and first_public is None: first_public = i
            if i < BOUNDARY:
                for row, tape in ((old, tapes[0]), (new, tapes[1])):
                    if row['decision']['requested_command'] != tape[i]['requested_command']:
                        raise ValueError('saved decision and executed request must agree')
    report = comparator.report() | commands | dict(physical_prefix_samples=PHYSICS_SAMPLES,
        original_physics_prefix_sha256=physics[0], candidate_physics_prefix_sha256=physics[1],
        physical_prefix_exact=physics[0] == physics[1], first_public_packet_difference=first_public,
        all_through_boundary_public_packets_exact=first_public is None,
        full_raw_sensor_audit_replaced=False, launch_and_model_admission_performed=False)
    report['budget_only_preboundary_execution_supported'] = bool(report['all_preboundary_decisions_exact']
        and commands['all_preboundary_commands_exact'] and physics[0] == physics[1] and first_public is None)
    for directory, ids in zip(directories, bindings, strict=True): verify_artifacts(directory.parent, ids)
    return report
