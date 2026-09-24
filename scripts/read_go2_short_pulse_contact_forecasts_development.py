"""Read unused contact forecasts on actually executed command prefixes."""
import json
import numpy as np
from lewm.geometry_progress_pilot_development import ACTIONS
from lewm.terminal_translation_pulse_development import command_sequences
from scripts.navigation_artifact_root_development import BASE

ARMS=('jepa','direct','supervised_rollout','pose_command')
OUTPUT=BASE/'go2_short_pulse_contact_forecasts_first_four_v1_attempt_002'


def probability(logit):
    return float(np.exp(-np.logaddexp(0.,-float(logit))))


def evaluate(arm):
    root=BASE/f'go2_short_pulse_navigation_{arm}_noise_2mm_native_layout00_4800_v1_attempt_001'
    read=lambda name:json.loads((root/name).read_text())
    if not (root/'short_pulse_navigation_evaluation_v1.json').exists():
        raise ValueError('completed and evaluated assignment required')
    with np.load(root/'native/physics_trace.npz',allow_pickle=False) as a:
        timestamps=np.rint(a['timestamp_s']*1e9).astype(np.int64)
        commands=a['requested_command'];contacts=a['physics_contact'].astype(bool)
    if not np.all(np.diff(timestamps)==2_000_000):raise ValueError('complete 2-ms physics required')
    plans=[p for p in read('planning.json') if 'selection' in p]
    rows=[];positive_plans={}
    for p in plans:
        now=p['measured_ns'];start=int(np.searchsorted(timestamps,now,side='right'))
        if contacts[:start].any():raise ValueError('post-contact prediction context')
        c=p['motion_correction'];index=ACTIONS.index(p['action'])
        tape=command_sequences(p['committed_prefix'],pulse=bool(c['terminal_translation_pulse']))[index]
        raw=np.asarray(c['upstream_prediction_for_yaw_ablation'])
        for h in range(1,9):
            end=now+h*100_000_000;stop=int(np.searchsorted(timestamps,end,side='right'))
            if stop<=start:continue
            ticks=(timestamps[start:stop]-now-1)//100_000_000
            # The physical runner records float32 command values in a float64
            # trace (e.g. 0.45 becomes 0.44999998807907104). Match that exact
            # representation instead of dropping turns through a tolerance.
            expected=tape[ticks].astype(np.float32).astype(commands.dtype)
            if not np.array_equal(commands[start:stop],expected):break
            event=bool(contacts[start:stop].any());complete=bool(timestamps[-1]>=end)
            if not complete and not event:continue
            forecast=probability(raw[index,h-1,4])
            rows.append(dict(frame=p['frame'],action=p['action'],horizon_ms=h*100,
                forecast_contact_probability=forecast,actual_contact=event,
                horizon_trace_complete=complete,positive_event_before_trace_ended=event and not complete))
            if event:
                positive_plans[p['frame']]=dict(frame=p['frame'],action=p['action'],
                    first_contact_offset_ms=float((timestamps[np.flatnonzero(contacts)[0]]-now)/1e6),
                    contact_probabilities_by_action={a:[probability(v) for v in raw[i,:,4]]
                        for i,a in enumerate(ACTIONS)},unexecuted_actions_not_evaluated=True)
    def metrics(population):
        if not population:return dict(windows=0)
        pr=np.array([r['forecast_contact_probability'] for r in population]);y=np.array([r['actual_contact'] for r in population])
        return dict(windows=len(population),positive_labels=int(y.sum()),
            brier_score=float(np.mean((pr-y)**2)),mean_probability=float(pr.mean()),
            maximum_probability=float(pr.max()),p95_probability=float(np.percentile(pr,95)),
            probability_at_least_half=int((pr>=.5).sum()))
    return dict(arm=arm,neural_reference_unused_for_motion=arm=='pose_command',
        native_contact_samples=int(contacts.sum()),unique_contact_events=int(np.count_nonzero(contacts & ~np.r_[False,contacts[:-1]])),
        all_horizons=metrics(rows),by_horizon={str(h):metrics([r for r in rows if r['horizon_ms']==h]) for h in range(100,801,100)},
        negative_labels=metrics([r for r in rows if not r['actual_contact']]),
        positive_labels=metrics([r for r in rows if r['actual_contact']]),
        positive_plans=list(positive_plans.values()),rows=rows)


def main():
    if OUTPUT.exists():raise ValueError('preserve previous readout')
    results=[evaluate(arm) for arm in ARMS];OUTPUT.mkdir()
    report=dict(status='COMPLETE',conditions=results,contact_head_disabled_in_all_live_runs=True,
        requested_commands_matched_after_exact_runner_float32_conversion=True,
        predecessor='go2_short_pulse_contact_forecasts_first_four_v1_attempt_001',
        predecessor_limitation='1e-8 comparison against unquantized commands excluded float32 turn requests; preserved but superseded',
        selected_executed_command_prefixes_only=True,native_contact_used_only_for_evaluation=True,
        positive_event_labels_retained_after_contact_stop=True,overlapping_horizons_not_independent=True,
        probability_calibration_established=False,alternate_navigation_policy_evaluated=False)
    with (OUTPUT/'result.json').open('x') as f:json.dump(report,f,indent=2)
    print(json.dumps([{k:v for k,v in r.items() if k not in ('rows','by_horizon')} for r in results]),flush=True)


if __name__=='__main__':main()
