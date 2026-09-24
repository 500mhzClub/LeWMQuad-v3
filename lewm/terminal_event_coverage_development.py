"""Prospective acquisition accounting, not task success or sensor qualification.

Use only after full raw sensor/contact attribution and stop reconstruction.
This layer independently checks coverage and censoring; it cannot replace that
audit with a caller-supplied success flag. No simulator geometry enters a model.
"""
from collections import Counter
import numpy as np
from lewm.independent_layout_collection_development import schedule

CONTACT_FIELDS = ('geom_a', 'geom_b', 'link_a', 'link_b', 'force_a', 'force_b', 'position', 'valid_mask')
PHYSICAL_REASONS = {'DISALLOWED_CONTACT', 'BODY_STABILITY_LIMIT',
                    'CONTEXT_NATIVE_CONTACT_SPEED_OR_DOMAIN_STOP'}
CLASSES = ('SCHEDULE_RECORDED', 'PHYSICAL_TERMINAL_RECORDED',
           'PRE_DEPARTURE_PHYSICAL_TERMINAL', 'SETUP_FAILED',
           'INFRASTRUCTURE_TRUNCATED', 'INVALID_RECORDING', 'UNATTEMPTED')


def require(condition, message):
    if not condition:
        raise ValueError(message)


def classify_coverage(spec, raw, contacts, events, cameras, tape, result,
                      report, prefix, window, labels):
    """Check one already raw-audited fixed-inventory recording, including stops."""
    n = len(raw['timestamp_s'])
    require(0 < n <= 2400 and n == result['physics_samples'] == report['physics_samples'],
            'complete bounded physics population required')
    require(all(len(v) == n and np.isfinite(v).all() for v in raw.values()), 'complete finite physics fields')
    ns = np.rint(raw['timestamp_s'] * 1e9).astype(np.int64)
    require(np.array_equal(ns, np.arange(1, n + 1) * 2_000_000), 'contiguous native clock required')
    contact = raw['physics_contact']
    require(contact.dtype == np.uint8 and contact.shape == (n,) and np.isin(contact, [0, 1]).all(),
            'binary native contact samples required')
    offsets = contacts['frame_offsets']
    require(offsets.shape == (n + 1,) and np.issubdtype(offsets.dtype, np.integer)
            and offsets[0] == 0 and (np.diff(offsets) >= 0).all(), 'complete contact offsets required')
    require(all(len(contacts[k]) == offsets[-1] for k in CONTACT_FIELDS), 'complete native contact fields required')
    require(np.array_equal(contacts['frame_timestamp_s'], raw['timestamp_s']), 'contact clock mismatch')
    require(len(events) == n, 'complete attributed contact events required')
    for i, event in enumerate(events):
        require(event['sample_index'] == i and event['timestamp_s'] == float(raw['timestamp_s'][i])
                and event['phase'] == int(raw['phase'][i])
                and bool(event['disallowed_contacts']) == bool(contact[i]), 'attributed contact coverage mismatch')

    for key in ('setup_checked', 'setup_admitted', 'departure_present'):
        require(type(result[key]) is bool and result[key] == report[key], 'setup/departure audit mismatch')
    require(report['recorded_sensor_reconstruction_pass'] is True, 'raw reconstruction required')
    stop, acquisition, terminal = result['physical_stop'], result['acquisition_stop'], result['schedule_terminal']
    require(sum(v is not None for v in (stop, acquisition, terminal)) == 1, 'exactly one terminal cause required')
    require(stop == report['physical_stop_message'] and acquisition == report['acquisition_stop'], 'terminal audit mismatch')
    require(report['schedule_complete'] == (terminal is not None), 'schedule audit mismatch')
    planned = schedule(spec['action_index'], spec['history_kind'])
    require(len(tape) == result['command_ticks'] <= len(planned), 'command population mismatch')
    require(result['completed_ticks'] == sum(t['completed'] for t in tape), 'completed command count mismatch')
    require(raw['requested_command'].dtype == np.float64, 'exact float64 command identity required')
    require(np.array_equal(raw['requested_command'][:min(n, 750)], np.zeros((min(n, 750), 3))), 'settling commands changed')
    for i, item in enumerate(tape):
        a, b = item['pre_sample_index'], item['post_sample_index']
        require(item['tick'] == i and a == 749 + 50 * i and type(b) is int
                and a <= b <= a + 50 and b < n and type(item['completed']) is bool, 'invalid command interval')
        require(all(item[k] == planned[i][k] for k in ('phase', 'role', 'requested_command')), 'planned command changed')
        require(np.array_equal(raw['requested_command'][a + 1:b + 1],
                               np.tile(np.asarray(planned[i]['requested_command'], np.float64), (b - a, 1))),
                'raw requested command changed')
        require((not item['completed'] or b == a + 50)
                and (i == len(tape) - 1 or item['completed']), 'incomplete nonterminal command')
    require(n == min(n, 750) + sum(t['post_sample_index'] - t['pre_sample_index'] for t in tape),
            'unaccounted or post-stop physics')
    require(not tape or tape[-1]['completed'] or stop is not None, 'interrupted command without physical stop')

    first = report['physical_stop']
    event_indices = np.flatnonzero(contact)
    if stop in PHYSICAL_REASONS:
        require(first == dict(sample_index=n - 1, reason=stop), 'exact first terminal sample required')
        require((event_indices.tolist() == [n - 1]) if stop == 'DISALLOWED_CONTACT' else len(event_indices) == 0,
                'contact reason or no-post-stop condition violated')
    elif stop == 'FRESH_MISSION_INITIAL_SETUP_REJECTED':
        require(first is None and n == 750 and result['setup_checked'] and not result['setup_admitted']
                and not len(event_indices), 'setup rejection mismatch')
    else:
        require(stop is None and first is None and not len(event_indices), 'unknown or missing physical terminal')
    if acquisition is not None:
        require(isinstance(acquisition, str) and (acquisition == 'STORAGE_RESERVE_STOP'
                or acquisition.startswith('PACKET_CONTRACT_STOP: ')), 'unknown acquisition cause')
    if terminal is not None:
        require(terminal == 'FIXED_CONTEXT_PULSE_COMPLETE' and result['setup_admitted']
                and len(tape) == len(planned) and all(t['completed'] for t in tape), 'incomplete claimed schedule')

    indices = [c['physical_sample_index'] for c in cameras]
    require(len(indices) == result['rgbd_frames'] == report['paired_frames'], 'camera population mismatch')
    if acquisition is not None:
        # Packet rejection may follow capture; storage rejection precedes it.
        choices = [list(range(749, 749 + 50 * count, 50)) for count in (len(tape), len(tape) + 1)]
        require(indices in choices, 'infrastructure capture coverage mismatch')
    else:
        count = len(planned) + 1 if terminal is not None else len(tape) if result['setup_admitted'] else int(result['setup_checked'])
        require(indices == [749 + 50 * i for i in range(count)], 'missing or post-terminal capture')
    require(all(type(i) is int and 0 <= i < n for i in indices), 'camera beyond recorded physics')
    require(all(c['timestamp_s'] == float(raw['timestamp_s'][i]) for c, i in zip(cameras, indices, strict=True)),
            'camera timestamp mismatch')
    departure = result['departure_present']
    require(departure == (len(tape) > 8), 'candidate action not actually attempted')
    if departure:
        require(result['setup_admitted'] and 1149 in indices and prefix['status'] == 'COMPLETE_PREFIX'
                and prefix['native_samples'] == 1150 and prefix['frames'] == 9 and bool(prefix['sha256']),
                'actual complete departure prefix required')
        require(window is not None and labels is not None and window['decision_ns'] == 2_300_000_000,
                'actual departure window and labels required')
        require(len(window['targets']) == len(labels['targets']) == 8, 'all target slots required')
        event_ns = int(ns[-1]) if stop == 'DISALLOWED_CONTACT' else None
        require(labels['accounting']['first_matched_contact_ns'] == event_ns
                and labels['accounting']['contact_before_departure'] is False, 'contact label accounting mismatch')
        for target, label in zip(window['targets'], labels['targets'], strict=True):
            offset = target['offset_ns']
            require(label['offset_ns'] == offset, 'target offset mismatch')
            known = offset > 0
            at = window['decision_ns'] + offset if known else None
            require(target['target_ns'] == at, 'target timestamp mismatch')
            end = 8 + offset // 100_000_000 if known else None
            executed = known and end <= len(tape) and all(t['completed'] for t in tape[8:end])
            observed = known and at // 2_000_000 - 1 in indices
            require(target['command_prefix_executed'] == bool(executed)
                    and target['observation_available'] == bool(observed)
                    and target['future_valid'] == bool(executed and observed), 'target acquisition coverage mismatch')
            event_seen = known and event_ns is not None and event_ns <= at
            endpoint = known and at <= int(ns[-1]) and target['command_prefix_executed']
            require(label['contact_valid'] == bool(endpoint or event_seen)
                    and label['motion_valid'] == bool(endpoint and not event_seen), 'invalid event or motion censoring')
            require(label['contact'] == (float(event_seen) if endpoint or event_seen else None), 'fabricated contact label')
            require(label['image_target_valid'] == target['future_valid'], 'future image mask mismatch')
            require(not target['future_valid'] or endpoint and target['observation_available'], 'post-stop image target')
            require(label['motion_valid'] or label['motion'] is None, 'fabricated censored motion')
    else:
        require(window is None and labels is None, 'pre-departure outcomes are not candidate-action targets')

    if acquisition is not None:
        status = 'INFRASTRUCTURE_TRUNCATED'
    elif not result['setup_admitted']:
        status = 'SETUP_FAILED'
    elif not departure:
        status = 'PRE_DEPARTURE_PHYSICAL_TERMINAL'
    elif terminal is not None:
        status = 'SCHEDULE_RECORDED'
    else:
        status = 'PHYSICAL_TERMINAL_RECORDED'
    return dict(classification=status, recording_accounted=True,
                candidate_action_observed=departure,
                candidate_acquisition_complete=status in ('SCHEDULE_RECORDED', 'PHYSICAL_TERMINAL_RECORDED'),
                command_schedule_complete=terminal is not None,
                terminal_reason=stop or acquisition or terminal, terminal_sample_index=n - 1,
                terminal_ns=int(ns[-1]), contact_event_count=int(len(event_indices)),
                physics_samples=n, paired_frames=len(cameras),
                strict_visibility_pass=report['physical_visibility_pass'],
                training_eligibility_granted=False, navigation_qualified=False)


def population_coverage(planned_ids, evidence, *, invalid=(), infrastructure=()):
    """Keep the full predeclared denominator; never infer success from absence."""
    planned = list(planned_ids)
    require(bool(planned) and len(set(planned)) == len(planned), 'unique nonempty planned population required')
    invalid, infrastructure = list(invalid), list(infrastructure)
    require(len(set(invalid)) == len(invalid) and len(set(infrastructure)) == len(infrastructure), 'duplicate failed case')
    groups = [set(evidence), set(invalid), set(infrastructure)]
    require(all(g <= set(planned) for g in groups) and sum(map(len, groups)) == len(set().union(*groups)),
            'unknown or overlapping case outcomes')
    statuses = {}
    for case in planned:
        status = (evidence[case]['classification'] if case in evidence else 'INVALID_RECORDING' if case in invalid
                  else 'INFRASTRUCTURE_TRUNCATED' if case in infrastructure else 'UNATTEMPTED')
        require(status in CLASSES, 'unknown recording classification')
        statuses[case] = status
    counts = Counter(statuses.values())
    return dict(planned_cases=len(planned), cases=statuses, counts={k: counts[k] for k in CLASSES},
                candidate_acquisition_complete=counts['SCHEDULE_RECORDED'] + counts['PHYSICAL_TERMINAL_RECORDED'],
                command_schedule_complete=counts['SCHEDULE_RECORDED'],
                training_eligibility_granted=False, navigation_qualified=False)
