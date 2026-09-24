"""Separately owned, capped historical re-render input check; zero physics.

    No reconstructed trajectories, replacement examples, or unqualified rendering
    adapters. Missing state for the qualified restore path is unresolved.
"""
from lewm import decision_headroom_json_v42_development as output_json
import argparse
import hashlib
import json
import os
from pathlib import Path
import resource
import shutil
import time
import psutil
from scripts.run_go2_decision_headroom_pilot_development import vram, retained_bytes

CONFIG=Path('docs/go2_decision_headroom_protocol_v42_2026-09-23.json')
BASE=Path('/home/andrewknowles/RecoveryStorage/LeWMQuad-v3/.generated/navigation_development_artifacts_v1')
CACHE=Path('/home/andrewknowles/.cache/lewm_go2_temporal_v03')


def safe(path):
    p=Path(path)
    if any(x=='sealed' or x=='sealed_test.json' or x.startswith('sealed_') for x in (*p.parts,*p.resolve().parts)):
        raise ValueError('protected path rejected')
    return p


def digest(path):
    with safe(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--approval',required=True,type=Path);args=parser.parse_args()
    config=json.loads(CONFIG.read_text());approval=json.loads(args.approval.read_text())
    assert approval['protocol_sha256']==digest(CONFIG) and config['historical_rerender_authorized'] is True
    caps=config['historical_rerender_proposal'];root=BASE/'go2_headroom_historical_rerender_v42_attempt_001'
    assert caps['physics_steps']==0 and caps['examples']==16 and caps['primary_images']==32
    root.mkdir(exist_ok=False);output_json.install(root);start=time.monotonic();cpu_start=time.process_time();owner=psutil.Process()
    measurements=[];results=[];error=None
    def save(name,value):
        payload=json.dumps(value,indent=2)+'\n'
        if retained_bytes(root)+len(payload.encode())>caps['retained_bytes']:raise RuntimeError('retained cap')
        (root/name).write_text(payload)
    def check():
        gpu=vram();usage=resource.getrusage(resource.RUSAGE_SELF)
        m=dict(wall_s=time.monotonic()-start,cpu_s=time.process_time()-cpu_start,
            rss_bytes=usage.ru_maxrss*1024,vram_bytes=gpu['used'],retained_bytes=retained_bytes(root),
            recovery_free_bytes=shutil.disk_usage(BASE).free,workspace_free_bytes=shutil.disk_usage(Path.cwd()).free)
        measurements.append(m)
        for key,cap in [('wall_s','wall_s'),('cpu_s','cpu_s'),('rss_bytes','ram_bytes'),('vram_bytes','vram_bytes'),('retained_bytes','retained_bytes')]:
            if m[key]>caps[cap]:raise RuntimeError('historical cap: '+key)
        if m['recovery_free_bytes']<caps['recovery_reserve_bytes'] or m['workspace_free_bytes']<caps['workspace_reserve_bytes']:raise RuntimeError('filesystem reserve')
    save('admission.json',dict(protocol_sha256=digest(CONFIG),approval_sha256=digest(args.approval),
        owner_pid=os.getpid(),owner_created=owner.create_time(),caps=caps,independent_of_phase2=True))
    try:
        check();inputs={};populations={}
        def load(path):
            path=safe(path);inputs[str(path)]=digest(path);return json.loads(path.read_text())
        fm=load(CACHE/'proprio_v1/factorial_manifest.json')
        populations['factorial_predictor_ancestor']=[dict(id=r['stable_row_id'],manifest_row=r) for r in fm['rows'] if r['split']=='train']
        native=Path('.generated/navigation_development_artifacts_v1/go2_horizon_dense_predictor_v1_attempt_001')
        paths=load(native/'frame_paths.json')
        populations['native_predictor_training']=[dict(id=f"{r['source']}/{r['trial']}/{r['frame']}/h{r['horizon']}",
            manifest_row={k:r[k] for k in ('source','trial','frame','horizon')},images=[paths[r['frames'][i]] for i in (2,3)]) for r in load(native/'samples.json')]
        for population,collection in [('full_heading_readout_training',Path('.generated/navigation_development_artifacts_v1/go2_full_heading_training_v1_attempt_001')),
                                      ('maze_view_readout_training',BASE/'go2_maze_view_training_v1_attempt_001')]:
            populations[population]=[dict(id=r['sample_id'],manifest_row={k:r[k] for k in ('sample_id','case','frame','data_role')},images=[r['current_rgb'],r['future_rgb']]) for r in load(collection/'samples.json') if r['data_role']=='train']
        selected=[]
        for population in caps['populations']:
            records=populations[population]
            assert len({r['id'] for r in records})==len(records)
            for r in records:r['priority']=hashlib.sha256(f"{caps['seed']}/{population}/{r['id']}".encode()).hexdigest()
            selected.extend(dict(r,population=population) for r in sorted(records,key=lambda r:(r['priority'],r['id']))[:4])
        assert len(selected)==16
        # Freeze identities before opening source images or reconstruction inputs.
        save('selection.json',dict(seed=caps['seed'],population_counts={k:len(v) for k,v in populations.items()},input_sha256=inputs,examples=selected))
        wanted={r['manifest_row']['pair_sha256']:r for r in selected if r['population']=='factorial_predictor_ancestor'}
        temporal=safe(CACHE/'temporal_rows.jsonl');inputs[str(temporal)]=digest(temporal)
        with temporal.open() as f:
            for line in f:
                r=json.loads(line)
                if r.get('role')=='train' and r.get('pair_sha256') in wanted:
                    wanted[r['pair_sha256']]['images']=[r['context_paths'][-1],r['target_path']]
        import numpy as np
        from PIL import Image
        for row in selected:
            check();frames=[]
            for name in row.get('images',[]):
                path=safe(name);frame=dict(path=str(path),exists=path.is_file())
                if path.is_file():
                    with Image.open(path) as im:array=np.asarray(im.convert('RGB'))
                    frame.update(file_sha256=digest(path),shape=list(array.shape),rgb_sha256=hashlib.sha256(array.tobytes()).hexdigest())
                directory=path.parent
                # Only direct recording metadata; no inventory crawl or sensor replay.
                names=('policy_observations.json','camera_metadata.json','specification.json','branch_specification.json',
                       'camera_terminal_identity.json','physics_topology.json','depth_retention.json','snapshots.json')
                frame['metadata_files']={n:dict(path=str(directory/n),sha256=digest(directory/n)) for n in names if (directory/n).is_file()}
                if (directory/'physics_trace.npz').is_file():
                    with np.load(directory/'physics_trace.npz',allow_pickle=False) as a:frame['physical_array_names']=list(a.files)
                frame['qualified_snapshot_index_present']=(directory/'snapshots.json').is_file()
                frame.update(status='unresolved',bitwise_equal=None,pixel_error_distribution=None,
                    reason='NO_RETAINED_BOUND_RESTORE_PACKET_FOR_QUALIFIED_PATH' if not frame['qualified_snapshot_index_present'] else 'RETAINED_INDEX_NEEDS_EXISTING_QUALIFIED_ADAPTER; NO_NEW_ADAPTER_OR_RECONSTRUCTION',
                    qualification_note='Pose/joint arrays and camera metadata alone are not the inventoried solver/scene/RNG packet consumed by the qualified restore. This does not prove later rendering impossible.')
                frames.append(frame)
            results.append(dict(population=row['population'],example_id=row['id'],priority=row['priority'],
                status='unresolved',source_frames=frames,rendered_frames=0,physics_steps=0,
                legacy_static_renderer=row['population']=='factorial_predictor_ancestor',
                reason='No qualified historical-to-current restoration adapter and bound full render state; no trajectory regenerated'))
            save('partial_result.json',results)
        check()
    except Exception as exc:
        error=repr(exc)
    finally:
        save('result.json',dict(status='COMPLETE_WITH_UNRESOLVED_INPUTS' if error is None else 'STOPPED',error=error,
            examples=results,examples_inspected=len(results),rendered_frames=0,physics_steps=0,
            training_render_provenance='render_unverified',phase2_launch_gate=False,retries=False,measurements=measurements))
    print(str(root/'result.json'))


if __name__=='__main__':main()
