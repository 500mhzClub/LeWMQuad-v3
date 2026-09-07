"""Prospective eight-run identity, explicit artifact roster and stream comparison."""
from scripts.independent_layout_batch_development import episode_artifacts
from scripts.run_go2_successive_choice_maze_development_v1 import digest
from scripts.audit_go2_independent_pulse_context_pilot_v1 import fingerprint
from scripts.startup_raw_sensor_audit_development import read_npz
from lewm.intent_return_rgbd_replay_development import IntentReturnRGBDReplay

TRIALS=('l00_junction_recent_forward_nominal_a0','l00_junction_recent_forward_nominal_a3',
        'l00_near_wall_recent_forward_lower_friction_a0','l00_near_wall_recent_forward_lower_friction_a1')
RUNS=tuple((f'repeat_{r}_{c}',c,r) for r in (0,1) for c in TRIALS)
RESERVE=40*1024**3
BUDGET=1024**3
ALLOWANCE=128*1024**2
PROTOCOL='docs/go2_ordered_union_dynamic_sensor_pilot_v1_2026-09-06.md'


def definitions(inventory):
    result={run:dict(trial=c,repeat=r,specification=inventory.specification(c)) for run,c,r in RUNS}
    if any(v['specification']['data_role']!='train' for v in result.values()):raise ValueError('training-role sensor diagnostic only')
    return result


def artifacts(spec,result):
    names=[n for n in episode_artifacts(spec,result) if not n.startswith('visual_meshes/')]
    names+=['visual_meshes/ground_visual.ply','visual_meshes/wall_union_visual.ply']
    names += [f'raster_{i:04d}.json' for i in range(result['rgbd_frames'])]
    return names


def commit_episode(output,run,spec,result):
    if (run,spec['trial']) not in [(a,b) for a,b,_ in RUNS]:raise ValueError('declared run/trial mapping required')
    names=[run+'/'+n for n in artifacts(spec,result)]
    present=[n for n in names if (output/n).is_file()]
    return dict(run=run,trial=spec['trial'],result=result,artifact_sha256={n:digest(output/n) for n in present},
        absent_expected_artifacts=sorted(set(names)-set(present)),artifact_bytes=sum((output/n).stat().st_size for n in present))


def whole_stream_witness(directory,result):
    raw=read_npz(directory,'physics_trace.npz');contacts=read_npz(directory,'native_contacts.npz')
    hashes={'native/'+k:fingerprint(v) for k,v in raw.items()}
    hashes.update({'contact/'+k:fingerprint(v) for k,v in contacts.items()})
    reader=IntentReturnRGBDReplay(directory) if result['rgbd_frames'] else None
    for i in range(result['rgbd_frames']):
        hashes['packet/'+str(i)]=fingerprint(reader.packet(i))
        hashes['depth/'+str(i)]=fingerprint(read_npz(directory,f'native_depth_{i:04d}.npz'))
    return dict(physics_samples=result['physics_samples'],frames=result['rgbd_frames'],
        schedule_complete=result['schedule_terminal']=='FIXED_CONTEXT_PULSE_COMPLETE',sha256=hashes)


def compare_streams(a,b):
    if a is None or b is None:return dict(status='UNAVAILABLE_STREAM',exact_available_stream=False,complete_repeatability=False,unequal_fields=[])
    unequal=sorted(k for k in set(a['sha256'])|set(b['sha256']) if a['sha256'].get(k)!=b['sha256'].get(k))
    for k in ('physics_samples','frames','schedule_complete'):
        if a[k]!=b[k]:unequal.append(k)
    exact=not unequal;complete=exact and a['schedule_complete'] and b['schedule_complete']
    return dict(status='EXACT_COMPLETE_REPLAY' if complete else 'EXACT_PARTIAL_REPLAY' if exact else 'UNEQUAL_REPLAY',
        exact_available_stream=exact,complete_repeatability=complete,unequal_fields=unequal)
