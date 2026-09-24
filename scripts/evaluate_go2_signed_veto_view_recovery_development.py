"""Evaluate physical navigation and auxiliary turns in signed-view recovery."""
from collections import Counter
from lewm.eligible_floor_registration_development import bind
from scripts import evaluate_go2_auxiliary_turn_recovery_development as previous
from scripts import run_go2_signed_veto_view_recovery_development as recovery


def diagnose(root):
    read = previous.previous.previous.read
    requests = read(root, 'requests.json')
    plans = read(root, 'planning.json')
    by_observation = {p['measured_ns']: p for p in plans}
    starts = {}
    grouped = {}
    for row in requests:
        state = row.get('view_recovery')
        if not state or state['source'] != 'actual_translation_veto':
            continue
        trigger = state['trigger_ns']
        starts.setdefault(trigger, row)
        grouped.setdefault(trigger, []).append(row)
    triggers = sorted(starts)
    events = []
    for i, trigger in enumerate(triggers):
        first = starts[trigger]
        state = first['view_recovery']
        last_ns = grouped[trigger][-1]['now_ns']
        next_trigger = triggers[i+1] if i+1 < len(triggers) else float('inf')
        vetoed = by_observation.get(state.get('vetoed_command_observation_ns',
            first.get('command_observation_ns')), {})
        command = state.get('vetoed_command', vetoed.get('selection', {}).get('requested_command'))
        recovery_plans = [p for p in plans if trigger <= p['measured_ns'] <= last_ns
            and p.get('route_status') == 'TRANSLATION_VETO_REQUIRES_NEW_VIEW']
        applied = Counter()
        for row in grouped[trigger]:
            cmd = row['applied_command']
            if not any(cmd[:2]) and cmd[2]:
                applied['right' if cmd[2] < 0 else 'left'] += 1
        translations = [r for r in requests if last_ns < r['now_ns'] < next_trigger
            and any(r['applied_command'][:2])]
        event = dict(trigger_ns=trigger, last_recovery_request_ns=last_ns,
            vetoed_action=vetoed.get('action'), vetoed_command=command,
            selected_view_angle_rad=state.get('view_angle_rad'),
            first_recovery_action=None if not recovery_plans else recovery_plans[0].get('action'),
            applied_turn_intervals=dict(applied),
            translation_intervals_after_recovery_before_next_veto=len(translations),
            first_translation_after_recovery_ns=None if not translations else translations[0]['now_ns'])
        events.append(event)
    return dict(root_name=root.name, translation_veto_recoveries=len(events), events=events,
        planning_actions=dict(Counter(p.get('action') for p in plans)),
        route_status_counts=dict(Counter(p.get('route_status') for p in plans)),
        translation_applied_intervals=sum(any(r['applied_command'][:2]) for r in requests),
        recoveries_followed_by_translation_before_next_veto=sum(
            e['translation_intervals_after_recovery_before_next_veto'] > 0 for e in events),
        applied_commands_are_not_measurements_of_physical_yaw=True)


if __name__ == '__main__':
    bind(previous.main, recovery=recovery)()
    root = previous.previous.previous.path(recovery.ROOT.format(index=1, arm='reactive'))
    result = diagnose(root)
    previous.previous.previous.save_or_read(root, 'signed_veto_view_recovery_diagnostic_v1.json', result)
    print({k:v for k,v in result.items() if k != 'events'}, flush=True)
