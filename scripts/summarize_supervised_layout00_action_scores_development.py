"""Describe every recorded decision of the completed supervised collection.

Reads saved scores and feasibility, without replay, inference or alternative
trajectory claims. The separately running raw audit remains authoritative.
"""
from collections import Counter
import gzip
import hashlib
import json
from pathlib import Path
import statistics
import time

import psutil

BASE = Path('/home/andrewknowles/RecoveryStorage/LeWMQuad-v3/.generated/navigation_development_artifacts_v1')
ROOT = BASE/'go2_stop_conditioned_independent_00_frozen_reference_seed_2026091001_full_supervised_rollout_v1_attempt_001'
CASE = ROOT/'independent_00_frozen_reference_seed_2026091001_full_supervised_rollout'
OUTPUT = Path('docs/go2_independent_layout00_supervised_complete_score_diagnosis_2026-09-13.json')
FAILURE = OUTPUT.with_suffix('.failure.json')
TRANSLATIONS = {'forward', 'left_arc', 'right_arc'}


def describe(values):
    if not values:
        return dict(count=0)
    return dict(count=len(values), minimum=min(values), maximum=max(values),
        mean=statistics.mean(values), median=statistics.median(values))


def main():
    if OUTPUT.exists() or FAILURE.exists():
        raise ValueError('preserve existing diagnosis')
    if psutil.virtual_memory().available < 16*1024**3:
        raise ValueError('retain RAM reserve for live audit')
    collection_bytes = (CASE/'result.json').read_bytes()
    collection = json.loads(collection_bytes)
    assert collection['decisions'] == 8014
    assert collection['schedule_terminal'] == 'MISSION_TICK_BUDGET_EXHAUSTED'
    counts = Counter(); actions = Counter(); contracts = Counter(); skips = Counter()
    feasible_sizes = Counter(); modes = Counter(); bins = {}
    gaps = []; progress_gains = []; contact_penalties = []
    first_translation_feasible_hold = None; last_translation_feasible_hold = None
    digest = hashlib.sha256(); started = time.perf_counter()
    try:
        with gzip.open(CASE/'context_decisions.jsonl.gz', 'rb') as stream:
            for frame, line in enumerate(stream):
                digest.update(line); row = json.loads(line)
                assert row['tick'] == frame
                decision = row['decision']; selection = decision['new_selection']
                action = str(decision['selected_action']); actions[action] += 1
                block = bins.setdefault(str(frame//1000), Counter())
                block[action] += 1; counts['decisions'] += 1
                command = decision['requested_command']
                counts['nonzero_translation_requests'] += bool(command[0] or command[1])
                counts['nonzero_yaw_requests'] += bool(command[2])
                if not selection:
                    skips['no_selection'] += 1; continue
                modes[str(selection.get('mode'))] += 1
                contracts[str(selection.get('score_contract'))] += 1
                if (selection.get('executed_waypoint_scoring') is not True
                        or selection.get('scored_pose_horizon_ns') != 100_000_000
                        or selection.get('scored_contact_horizon_ns') != 800_000_000):
                    skips['outside_executed_waypoint_100ms_800ms_score'] += 1; continue
                candidates = selection['candidates']; counts['analyzed_score_banks'] += 1
                feasible = [i for i, c in enumerate(candidates)
                    if c['action'] in selection['phase_allowed_actions']
                    and not selection['surface_checks'][i]['possible_intersection']
                    and selection['nominal_path_checks'][i]['all_predicted_segments_nominally_clear']]
                feasible_sizes[str(len(feasible))] += 1
                progress = [c['executed_waypoint_distance_progress_m']+
                    c['executed_waypoint_alignment_progress_m'] for c in candidates]
                penalty = selection['contact_penalty_m']
                utilities = [p-penalty*c['full_plan_contact_score']
                    for p, c in zip(progress, candidates, strict=True)]
                assert all(abs(u-c['utility_m']) <= 1e-12 for u, c in zip(utilities, candidates, strict=True))
                if not feasible:
                    counts['empty_feasible_banks'] += 1; continue
                best = max(feasible, key=lambda i: utilities[i])
                if candidates[best]['action'] != selection['action']:
                    counts['selection_differs_from_score_maximum'] += 1; continue
                counts['reconstructed_final_selections'] += 1
                available = {candidates[i]['action']:i for i in feasible}
                if TRANSLATIONS & available.keys():
                    counts['translation_feasible'] += 1
                    if selection['action'] == 'hold':
                        counts['hold_with_translation_feasible'] += 1
                        block['hold_with_translation_feasible'] += 1
                        if first_translation_feasible_hold is None:
                            first_translation_feasible_hold = frame
                        last_translation_feasible_hold = frame
                if selection['action'] == 'hold' and 'forward' in available:
                    h, f = available['hold'], available['forward']
                    gain = progress[f]-progress[h]
                    contact = penalty*(candidates[f]['full_plan_contact_score']-candidates[h]['full_plan_contact_score'])
                    gaps.append(utilities[h]-utilities[f]); progress_gains.append(gain); contact_penalties.append(contact)
                    counts['forward_progress_advantage_outweighed_by_contact'] += gain > 0 and contact > gain
                if frame%1000 == 0:
                    print('SUPERVISED_SCORE_FRAME', frame, flush=True)
        assert counts['decisions'] == collection['decisions']
        assert (CASE/'result.json').read_bytes() == collection_bytes
        report = dict(status='COMPLETE_RECORDED_SUPERVISED_SCORE_DIAGNOSIS',
            counts=dict(counts), selected_actions=dict(actions), score_contracts=dict(contracts),
            modes=dict(modes), skipped=dict(skips), feasible_bank_sizes=dict(feasible_sizes),
            thousand_frame_bins={k:dict(v) for k,v in bins.items()},
            first_translation_feasible_hold_frame=first_translation_feasible_hold,
            last_translation_feasible_hold_frame=last_translation_feasible_hold,
            hold_over_forward_utility_gap_m=describe(gaps),
            forward_over_hold_progress_gain_m=describe(progress_gains),
            forward_over_hold_contact_penalty_m=describe(contact_penalties),
            collection_sha256=hashlib.sha256(collection_bytes).hexdigest(),
            decoded_decision_stream_sha256=digest.hexdigest(), input=str(CASE),
            wall_s=time.perf_counter()-started, native_execution=False,
            model_inference_executed=False, controller_replay_executed=False,
            alternative_trajectory_inferred=False, navigation_improvement_established=False,
            full_raw_audit_pending_at_preparation=True)
        with OUTPUT.open('x') as out:
            json.dump(report, out, indent=2, allow_nan=False); out.write('\n')
        print(json.dumps(dict(counts=dict(counts), actions=dict(actions),
            skipped=dict(skips), wall_s=report['wall_s'])), flush=True)
    except BaseException as error:
        with FAILURE.open('x') as out:
            json.dump(dict(reason=repr(error), counts=dict(counts)), out, indent=2)
        raise


if __name__ == '__main__':
    main()
