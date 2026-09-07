"""Bound full recorded dataset materialization; no training/evaluation claim."""
import hashlib
import json
from collections import Counter
from lewm.pulse_timed_dataset_development import PulseTimedDataset
from lewm.intent_return_rgbd_replay_development import IntentReturnRGBDReplay
from scripts.check_go2_pulse_timed_pairing_v1 import INPUT,OUTPUT as PAIRING,TRIALS
from scripts.derive_go2_recorded_pulse_native_targets_v1 import OUTPUT as LABELS
from scripts.startup_raw_sensor_audit_development import read_json
from scripts.startup_source_inventory_development import discover_sources
from scripts.analyze_go2_ground_plane_development_v1 import verify_bindings
from scripts.run_go2_successive_choice_maze_development_v1 import ROOT,digest,write_json

OUTPUT=ROOT/'.generated/go2_pulse_dataset_diagnostic_v1_attempt_001'
PROTOCOL='docs/go2_pulse_dataset_diagnostic_v1_2026-09-06.md'


def preflight():
    ids={str((LABELS/n).relative_to(ROOT)):h for n,h in {
        'launch.json':'361376225ffc4bd67f7ebcd6eb25fd3dc39de52dca3967e3100e7e10a7b3ad20',
        'result.json':'10418dec703a41947bc47b1f5f1851069ed376e5f6c4f1565cff7899fe59a4fe',
        'targets.json':'a5dde045d15b7fd1269a147a24ed1decbc379b1a09ea0cdfb0f1e268d20e6b0f'}.items()}
    verify_bindings(ids);old=read_json(LABELS,'launch.json');verify_bindings(old['source_sha256']|old['input_sha256'])
    collection=read_json(INPUT,'result.json')
    if collection['absent_expected_artifacts']:raise ValueError('complete raw collection required')
    inputs=old['input_sha256']|ids|{str((INPUT/n).relative_to(ROOT)):h for n,h in collection['artifact_sha256'].items()}
    sources=discover_sources((PROTOCOL,'scripts/check_go2_pulse_dataset_v1.py',
        'lewm/tests/test_pulse_timed_dataset_development.py'),old['source_sha256'])
    verify_bindings(sources|inputs)
    return dict(source_sha256=sources,input_sha256=inputs,model_training=False,new_physics=False,
        scope='all185 old pulse windows, one room layout, train-role development plumbing only',goal_achieved=False)


def main():
    if OUTPUT.exists() or OUTPUT.is_symlink():raise ValueError('exclusive dataset diagnostic')
    launch=preflight();OUTPUT.mkdir();write_json(OUTPUT/'launch.json',launch)
    try:
        specs={c:read_json(INPUT/c,'specification.json') for c in TRIALS}
        walls=specs[TRIALS[0]]['geometry']['wall_boxes']
        if any(s['geometry']['wall_boxes']!=walls for s in specs.values()):raise ValueError('expected common room layout')
        layout='same-controlled-four-wall-room-'+hashlib.sha256(json.dumps(walls,sort_keys=True).encode()).hexdigest()
        roles={c:dict(layout_id=layout,role='train') for c in TRIALS}
        dataset=PulseTimedDataset(read_json(PAIRING,'windows.json'),read_json(LABELS,'targets.json'),roles)
        schedules={arm:dataset.schedule('train',updates=12,batch_size=6,seed=2026090711)
                   for arm in ('direct','supervised_rollout','jepa')}
        if len({s['schedule_sha256'] for s in schedules.values()})!=1:raise ValueError('matched schedules required')
        readers={c:IntentReturnRGBDReplay(INPUT/c) for c in TRIALS};counts=Counter()
        for i in range(len(dataset)):
            if not dataset.windows[i]['history_ready']:continue
            sample=dataset.sample(i,readers);t=sample['targets']
            counts.update(dict(materialized_windows=1,motion_valid=int(t['motion_valid'].sum()),
                contact_valid=int(t['contact_valid'].sum()),positive_contacts=int(t['contact'][t['contact_valid']].sum()),
                future_images=int(t['future_valid'].sum())))
            if i%25==0:print('PULSE_DATASET',i,dict(counts),flush=True)
        verify_bindings(launch['source_sha256']|launch['input_sha256'])
        write_json(OUTPUT/'schedules.json',schedules)
        write_json(OUTPUT/'result.json',dict(status='PULSE_DATASET_MATERIALIZATION_COMPLETE',
            windows=len(dataset),counts=dict(counts),excluded=dataset.excluded,
            episode_roles=roles,coverage=dataset.coverage('train'),independent_layouts=1,
            selection_layouts=0,evaluation_layouts=0,schedules_sha256=digest(OUTPUT/'schedules.json'),
            schedule_identity=schedules['jepa']['schedule_sha256'],optimizer_steps=0,
            model_training=False,maze_generalization_established=False,goal_achieved=False))
    except Exception as error:
        write_json(OUTPUT/'failure.json',dict(status='TERMINAL_DATASET_DIAGNOSTIC_FAILURE',reason=repr(error)));raise


if __name__=='__main__':main()
