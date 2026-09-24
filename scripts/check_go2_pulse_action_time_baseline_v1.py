"""Posthoc action/time control on frozen one-room training draws; no navigation."""
import shutil
import numpy as np
from lewm.pulse_action_time_baseline_development import ActionTimeMean
from lewm.pulse_timed_dataset_development import PulseTimedDataset, decode_targets, key
from lewm.pulse_timed_rgb_body_jepa_development import pulse_brake_plan, validate_timed_plan
from lewm.temporal_prediction_metrics_development import reduce_predictions
from scripts.run_go2_pulse_training_pilot_v2 import OUTPUT as PILOT
from scripts.check_go2_pulse_dataset_v1 import OUTPUT as DATA
from scripts.check_go2_pulse_timed_pairing_v1 import OUTPUT as PAIRING
from scripts.derive_go2_recorded_pulse_native_targets_v1 import OUTPUT as LABELS
from scripts.startup_raw_sensor_audit_development import read_json
from scripts.startup_source_inventory_development import discover_sources
from scripts.analyze_go2_ground_plane_development_v1 import verify_bindings
from scripts.run_go2_successive_choice_maze_development_v1 import ROOT, digest, write_json

OUTPUT = ROOT/'.generated/go2_pulse_action_time_baseline_v1_attempt_001'
PROTOCOL = 'docs/go2_pulse_action_time_baseline_v1_2026-09-06.md'
RESERVE = 10*1024**3


def preflight():
    ids = {str((PILOT/n).relative_to(ROOT)):h for n,h in {
        'launch.json':'5059f97d20817d8781f1ecdabe10d811ad5e0741a2ca96861985a58a84e21984',
        'result.json':'ecfbd5b84ac65821134c28b7983230c8f9196dce92325c1e4b2f0c683e15e032'}.items()}
    verify_bindings(ids)
    old = read_json(PILOT, 'launch.json')
    verify_bindings(old['source_sha256'] | old['input_sha256'])
    sources = discover_sources((PROTOCOL, 'scripts/check_go2_pulse_action_time_baseline_v1.py',
        'lewm/tests/test_pulse_action_time_baseline_development.py'), old['source_sha256'])
    inputs = old['input_sha256'] | ids
    verify_bindings(sources | inputs)
    if shutil.disk_usage(ROOT).free < RESERVE+10*1024**2:
        raise ValueError('10MiB diagnostic budget plus10GiB reserve required')
    return dict(source_sha256=sources, input_sha256=inputs, numpy_version=np.__version__,
        posthoc=True, resubstitution=True, new_physics=False, neural_training=False,
        navigation_qualified=False, independent_generalization_established=False, goal_achieved=False)


def arrays(dataset, rows):
    """Plan-only query construction, separately joined native scoring targets."""
    lookup = {key(r):r for r in rows}
    actions, times, masks, native, metadata = [], [], [], [], []
    for w in dataset.windows:
        if not w['history_ready']:
            raise ValueError('this fixed diagnostic requires all185 existing eligible windows')
        blocks, valid = pulse_brake_plan(tuple(w['command']), w['pulse_ticks'])
        active, offsets = validate_timed_plan(blocks[None], valid[None], 1)
        if not np.array_equal(active[0].numpy(), valid.any(-1).numpy()):
            raise ValueError('partial-block query contract')
        actions.append(w['action_index']); times.append(offsets[0].numpy()); masks.append(active[0].numpy())
        native.append(decode_targets(w, lookup[key(w)]))
        metadata.append(dict(dataset.episode_roles[w['condition']], condition=w['condition'],
            action_index=w['action_index']))
    targets = {k:np.stack([n[k].numpy() for n in native])
        for k in ('motion','contact','motion_valid','contact_valid')}
    return np.array(actions), np.stack(times), np.stack(masks), targets, metadata


def main():
    if OUTPUT.exists() or OUTPUT.is_symlink():
        raise ValueError('exclusive diagnostic; no implicit retry or overwrite')
    launch = preflight(); OUTPUT.mkdir(); write_json(OUTPUT/'launch.json', launch)
    try:
        data = read_json(DATA, 'result.json'); schedules = read_json(DATA, 'schedules.json')
        rows = read_json(LABELS, 'targets.json')
        dataset = PulseTimedDataset(read_json(PAIRING, 'windows.json'), rows, data['episode_roles'])
        schedule = dataset.schedule('train', updates=12, batch_size=6, seed=2026090711)
        if any(schedules[arm] != schedule for arm in ('direct','supervised_rollout','jepa')):
            raise ValueError('exact shared frozen training schedule required')
        a,t,v,y,metadata = arrays(dataset, rows)
        ids = [i for batch in schedule['batches'] for i in batch]
        if len(a)!=185 or len(ids)!=72 or len(set(ids))!=62:
            raise ValueError('fixed corpus and draw counts required')
        model = ActionTimeMean.fit(a[ids],t[ids],v[ids],{k:x[ids] for k,x in y.items()},
            roles=[metadata[i]['role'] for i in ids])
        write_json(OUTPUT/'table.json', model.record())
        write_json(OUTPUT/'schedule.json', schedule)
        pred, missing = model.predict(a,t,v)
        zero = np.zeros_like(pred); zero[...,3]=1; zero[...,4]=-30
        def score(p):
            return dict(all=reduce_predictions(p,y,metadata,v),
                by_actual_offset_ns={str(int(ns)):reduce_predictions(p,y,metadata,v,horizon_selection=t==ns)
                    for ns in sorted(set(t[v]))},
                by_action={str(action):reduce_predictions(p,y,metadata,v,row_selection=a==action)
                    for action in range(6)},
                by_condition={c:reduce_predictions(p,y,metadata,v,
                    row_selection=np.array([m['condition']==c for m in metadata]))
                    for c in sorted({m['condition'] for m in metadata})})
        metrics = {'zero_motion_no_contact':score(zero)}
        if not missing:
            metrics['action_time_mean'] = score(pred)
            write_json(OUTPUT/'predictions.json', dict(actions=a.tolist(),offsets_ns=t.tolist(),
                active=v.tolist(),prediction=pred.tolist()))
        verify_bindings(launch['source_sha256'] | launch['input_sha256'])
        artifacts = ['table.json','schedule.json'] + ([] if missing else ['predictions.json'])
        result = dict(status='COMPLETE' if not missing else 'COMPLETE_WITH_UNSUPPORTED_CELLS',
            launch_sha256=digest(OUTPUT/'launch.json'), missing_cells=missing, metrics=metrics,
            artifact_sha256={n:digest(OUTPUT/n) for n in artifacts},
            training_draws=len(ids),distinct_training_windows=len(set(ids)),scored_windows=len(a),
            distinct_layouts=len({m['layout_id'] for m in metadata}),role='train',
            resubstitution=True,posthoc=True,neural_training=False,navigation_qualified=False,
            independent_generalization_established=False,goal_achieved=False)
        write_json(OUTPUT/'result.json',result)
        print({k:x['all']['layout_macro'] for k,x in metrics.items()},flush=True)
    except Exception as error:
        write_json(OUTPUT/'failure.json',dict(status='TERMINAL_FAILURE',reason=repr(error)))
        raise


if __name__ == '__main__': main()
