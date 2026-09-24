"""User-amended two-state/42-branch renderer restoration recheck; no audit rows."""
import contextlib
import hashlib
import itertools
import json
import math
import os
from pathlib import Path
import shutil
import tempfile
import time
import traceback

import cv2
import numpy as np
from PIL import Image
import psutil
import torch

from scripts import run_go2_dense_horizon_navigation_development as native
from scripts.run_go2_decision_headroom_branches_development import (
    ACTIONS, TRACE_FIELDS, compare_trace, execute, load_bound, save,
)
from scripts.run_go2_decision_headroom_pilot_development import PilotBudget, REPO, GIB, vram
from lewm.decision_headroom_reference_development import ReferenceGeometry


MANIFEST = Path('docs/go2_decision_headroom_rgb_recheck_amended_v2_2026-09-23.json')


class RecheckBudget(PilotBudget):
    def reserve_branch(self, identity):
        if len(self.branches) >= len(self.schedule) or identity != self.schedule[len(self.branches)]:
            raise ValueError('fixed amended branch order and unique identities required')
        self.check('branch_admission', force=True)
        self.branches.add(identity)
        self.event('branch_reserved', identity=identity, physics_ns=800_000_000)


def pixel_comparison(first, second):
    a, b = np.asarray(Image.open(first)), np.asarray(Image.open(second))
    if a.shape != (480,640,3) or b.shape != a.shape or a.dtype != np.uint8 or b.dtype != np.uint8:
        raise ValueError('unchanged native lossless RGB required')
    difference = np.abs(a.astype(np.int16)-b.astype(np.int16))
    return dict(bitwise_equal=bool(np.array_equal(a,b)), mae=float(difference.mean()),
        rmse=float(np.sqrt(np.mean(difference.astype(float)**2))),
        channel_error_quantiles=np.quantile(difference,[0,.5,.95,.99,1]).tolist(),
        changed_pixel_fraction=float(np.any(difference,axis=2).mean()),
        absolute_channel_error_histogram=np.bincount(difference.ravel(),minlength=256).tolist())


def recorded_camera_rows(state):
    root = Path(state['source_root'])
    rows = json.loads((root/'native/in_memory_camera_observations.json').read_text())['frames']
    return {r['measured_ns']:r for r in rows}


def source_image_path(state, row, label):
    prefix = 'rgb' if label=='primary' else 'auxiliary_rgb'
    return Path(state['source_root'])/'native'/f"{prefix}_{row['frame']:04d}.png"


def collect_state(state, index, manifest, root, budget):
    source = Path(state['source_root']); metadata=state['snapshot']
    frozen = source/f"state_{state['frame']:04d}"
    snapshot = load_bound(frozen/'physical.pkl', metadata['physical'])
    decision = load_bound(frozen/'decision.pkl', metadata['decision'])
    assert tuple(decision['candidate_order']) == ACTIONS
    stamp = state['measured_ns']
    requested = json.loads((source/'requests.json').read_text())
    recorded = [r for r in requested if stamp <= r['simulator_ns'] < stamp+800_000_000]
    assert [r['simulator_ns'] for r in recorded] == list(range(stamp,stamp+800_000_000,20_000_000))
    source_commands = np.asarray([r['applied_command'] for r in recorded])
    with np.load(source/'native/physics_trace.npz',allow_pickle=False) as archive:
        truth = {k:archive[k].copy() for k in TRACE_FIELDS}
    cameras = recorded_camera_rows(state)
    spec = json.loads((source/'specification.json').read_text())
    directory = root/f'state_{index:02d}';directory.mkdir()
    scene_output=directory/'native_setup';scene_output.mkdir()
    walls=[dict(center=w['centre_xyz'][:2],size=w['size_xyz'][:2],yaw=w['yaw_rad']) for w in spec['geometry']['wall_boxes']]
    extents=np.array([w['center'] for w in walls])
    geometry=ReferenceGeometry(walls,[extents.min(axis=0)-2,extents.max(axis=0)+2],
        spec['geometry']['spawn_se2_world'][:2],radius_m=.46,clearance_m=.005,resolution_m=.02)
    session=None;comparisons=[];previous={};start=time.monotonic()
    try:
        native.initialize_genesis(backend='cpu',seed=spec['procedural_seed'],logging_level='warning')
        session=native.ProspectiveRGBOnlySession(spec,scene_output,noise_layout_index=0,noise_sigma_mm=2)
        session.install_contact_identity()
        save(directory/'actuator_identity.json',native.configure_gains(session.ctx.build.robot,
            session.ctx.runner._leg_dof_idx.tolist(),session.ctx.policy.env_cfg,'checkpoint'))
        budget.check('before_settling',force=True)
        budget.source_ns[index]=1_500_000_000
        budget.event('scene_settling_reserved',state=index,physics_ns=1_500_000_000)
        session.settle_recorded()
        native.admit_context_setup(session,hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
        session.physics_clock_callback=None
        for name in manifest['branch_order_per_state']:
            is_source=name.startswith('source_trace_')
            if is_source:
                tape=projected=source_commands
            else:
                action,repeat=name.rsplit('_',1);action_index=ACTIONS.index(action)
                tape=np.repeat(decision['candidate_requested_commands'][action_index],5,axis=0)
                projected=np.repeat(decision['candidate_applied_commands'][action_index],5,axis=0)
            output=directory/name
            actual,terminal=execute(session,snapshot,tape,output,budget,
                identity=f'state_{index:02d}/{name}',geometry=geometry,expected_applied=projected)
            frames=json.loads((output/'frames.json').read_text())
            comparisons_for_branch=[]
            if is_source:
                physics=compare_trace(actual,truth,stamp,manifest['physical_tolerances'])
                pixels=[]
                for frame in frames:
                    recorded_frame=cameras[stamp+frame['offset_ms']*1_000_000]
                    for j,label in enumerate(('primary','auxiliary')):
                        path=source_image_path(state,recorded_frame,label)
                        raw=np.asarray(Image.open(path))
                        if hashlib.sha256(raw.tobytes()).hexdigest()!=recorded_frame['pixel_sha256'][label]['rgb_sha256']:
                            raise ValueError('historical source pixels changed')
                        pixels.append(dict(offset_ms=frame['offset_ms'],camera=label,
                            camera_transform_maximum_error=float(np.max(np.abs(np.asarray(frame['transforms'][j])-recorded_frame['transforms'][j]))),
                            **pixel_comparison(path,output/frame['images'][label]['path'])))
                comparisons_for_branch.append(dict(comparator='original_source',physics=physics,pixels=pixels))
            else:
                for prior_name,prior_arrays,prior_frames in previous.setdefault(action,[]):
                    physics=compare_trace(actual,prior_arrays,stamp,manifest['physical_tolerances'])
                    pixels=[]
                    for frame,prior in zip(frames,prior_frames,strict=True):
                        assert frame['offset_ms']==prior['offset_ms']
                        for label in ('primary','auxiliary'):
                            pixels.append(dict(offset_ms=frame['offset_ms'],camera=label,
                                **pixel_comparison(directory/prior_name/prior['images'][label]['path'],output/frame['images'][label]['path'])))
                    comparisons_for_branch.append(dict(comparator=prior_name,physics=physics,pixels=pixels))
                previous[action].append((name,actual,frames))
            row=dict(branch=name,is_source_trace=is_source,terminal=terminal,
                comparisons=comparisons_for_branch,complete=len(frames)==8 and terminal is None,
                source_replay_after_candidate_branches=is_source and name!='source_trace_0')
            save(output/'restoration_comparison.json',row);comparisons.append(row)
        save(directory/'result.json',dict(source_state=state,branches=comparisons,wall_s=time.monotonic()-start))
        return comparisons
    finally:
        if session is not None:session.ctx.build.scene.destroy()
        native.shutdown_genesis()


@torch.inference_mode()
def feature_diagnostics(manifest,root,budget):
    from scripts.dev_frozen_dense_representation_encoders_v1 import VJepa21Arm, _normalise, _to_chw
    from torch.nn import functional as F
    remaining=manifest['limits']['execution_wall_s']-(time.monotonic()-budget.started)
    gpu=vram()
    if remaining<180 or gpu['used']+4*GIB>manifest['limits']['total_gpu_used_bytes']:
        return dict(status='OMITTED',reason='insufficient declared wall/VRAM headroom for optional encoder diagnostics',encoded_images=0)
    encoder=VJepa21Arm();encoder.build(torch.device('cuda:0'),torch.float32)
    budget.check('encoder_loaded',force=True)
    count=0;rows=[]
    def encode(path):
        nonlocal count
        budget.check('before_optional_encoding',force=True)
        if count>=manifest['optional_features']['max_encoder_images']:raise RuntimeError('optional encoder image cap')
        pixels=_normalise(_to_chw(Image.open(path).convert('RGB').resize((512,384),Image.Resampling.BICUBIC)))[None].cuda()
        count+=1
        encoded=encoder.tokens(pixels).float().cpu()
        if not torch.isfinite(encoded).all():raise ValueError('nonfinite encoder diagnostic')
        torch.cuda.synchronize();budget.check('after_optional_encoding',force=True)
        return encoded
    def differences(a,b):
        normal_a,normal_b=F.layer_norm(a,(1024,)),F.layer_norm(b,(1024,))
        mse=float((a-b).square().mean());normal_mse=float((normal_a-normal_b).square().mean())
        return dict(raw_mse=mse,raw_rmse=math.sqrt(mse),normalised_mse=normal_mse,
            normalised_rmse=math.sqrt(normal_mse),cosine_distance=float(1-F.cosine_similarity(a.flatten(),b.flatten(),dim=0)))
    for index,state in enumerate(manifest['states']):
        cameras=recorded_camera_rows(state);directory=root/f'state_{index:02d}'
        for ms in range(100,801,100):
            if manifest['limits']['execution_wall_s']-(time.monotonic()-budget.started)<45:
                return dict(status='PARTIAL',reason='reserved closeout wall time',encoded_images=count,rows=rows)
            source=encode(source_image_path(state,cameras[state['measured_ns']+ms*1_000_000],'primary'))
            replays=[encode(directory/f'source_trace_{r}'/f'primary_{ms:03d}ms.png') for r in range(3)]
            candidates={a:encode(directory/f'{a}_0'/f'primary_{ms:03d}ms.png') for a in ACTIONS}
            between=[dict(candidate_a=a,candidate_b=b,**differences(candidates[a],candidates[b])) for a,b in itertools.combinations(ACTIONS,2)]
            denominator={key:float(np.mean([x[key] for x in between])) for key in ('raw_mse','normalised_mse')}
            replay=[]
            for repeat,feature in enumerate(replays):
                diff=differences(feature,source);ratios={}
                for key,value in denominator.items():
                    ratios[key]=None if value<=manifest['optional_features']['negligible_between_candidate_mse'] else diff[key]/value
                replay.append(dict(repeat=repeat,**diff,ratio_to_mean_between_candidate_mse=ratios))
            rows.append(dict(state=index,horizon_ms=ms,shared_prefix=ms<=300,replay_source=replay,
                between_candidates=between,mean_between_candidate_mse=denominator,
                ratio_undefined_threshold=manifest['optional_features']['negligible_between_candidate_mse']))
            del source,replays,candidates
    return dict(status='COMPLETE',encoded_images=count,rows=rows,rankings_computed=False,
        dense_features_retained=False,visual_discrepancy_harmlessness_established=False)


def main():
    manifest=json.loads(MANIFEST.read_text());root=Path(manifest['output_root']);limits=manifest['limits']
    for name,digest in manifest['source_sha256'].items():
        if hashlib.sha256(Path(name).read_bytes()).hexdigest()!=digest:raise RuntimeError('frozen recheck source changed')
    if root.exists():raise RuntimeError('existing recheck preserved; no restart')
    for p,reserve,peak in ((root.parent,limits['recovery_reserve_bytes'],limits['peak_additional_bytes']),
                           (REPO,limits['workspace_reserve_bytes'],512*1024**2)):
        if shutil.disk_usage(p).free < reserve+peak:raise RuntimeError('filesystem admission')
    for binding in manifest['input_bindings'].values():
        with Path(binding['path']).open('rb') as stream:
            if hashlib.file_digest(stream,'sha256').hexdigest()!=binding['sha256']:raise RuntimeError('frozen recheck input changed')
    if psutil.virtual_memory().available<32*GIB:raise RuntimeError('RAM admission')
    if vram()['used']>limits['total_gpu_used_bytes']-4*GIB:raise RuntimeError('VRAM admission')
    os.sched_setaffinity(0,set(range(8,16))|set(range(24,32)));cv2.setNumThreads(1);torch.set_num_threads(4)
    root.mkdir();(root/'scratch/temp').mkdir(parents=True)
    os.environ['TMPDIR']=str(root/'scratch/temp');tempfile.tempdir=str(root/'scratch/temp')
    caps=dict(compute_caps=dict(execution_wall_seconds=limits['execution_wall_s'],aggregate_cpu_seconds=limits['aggregate_cpu_s'],
        aggregate_rss_bytes=limits['aggregate_rss_bytes'],gpu_allocated_bytes=limits['total_gpu_used_bytes'],
        minimum_available_ram_bytes=16*GIB,minimum_available_vram_bytes=4*GIB),
        storage_caps=dict(retained_bytes=limits['retained_bytes'],peak_additional_bytes=limits['peak_additional_bytes'],
            recovery_filesystem_reserve_bytes=limits['recovery_reserve_bytes'],workspace_filesystem_reserve_bytes=limits['workspace_reserve_bytes']))
    admission=dict(owner_pid=os.getpid(),owner_created=psutil.Process().create_time(),
        manifest_sha256=hashlib.sha256(MANIFEST.read_bytes()).hexdigest(),manifest=manifest,
        cache_paths=[str(Path.home()/'.cache'/name) for name in ('genesis','quadrants','gstaichi','triton','torch','mesa_shader_cache','mesa_shader_cache_db')]+
            [str(REPO/'.generated/box_meshes'),'/tmp/torchinductor_'+Path.home().name],
        paths=dict(output=str(root),temporary=str(root/'scratch/temp'),source_inputs=[s['source_root'] for s in manifest['states']],
            encoder_checkpoint=str(Path.home()/'.cache/vjepa2_1_vitl_dist_vitG_384.pt'),encoder_source=str(Path.home()/'.cache/vjepa2-204698b45b3712590f06245fbfba32d3be539812')),
        filesystems={str(p):dict(device=p.stat().st_dev,available_bytes=shutil.disk_usage(p).free) for p in (root,REPO,Path.home()/'.cache',Path('/tmp'))})
    save(root/'admission.json',admission)
    budget=RecheckBudget(root,caps,admission)
    budget.schedule=[f'state_{i:02d}/{name}' for i in range(2) for name in manifest['branch_order_per_state']]
    error=None;states=[];features=None
    try:
        native.previous.warmup();native.previous.study.cohort.stable.floor.configure()
        with (root/'worker.log').open('x') as log,contextlib.redirect_stdout(log),contextlib.redirect_stderr(log):
            for index,state in enumerate(manifest['states']):
                states.append(collect_state(state,index,manifest,root,budget))
                print('RECHECK_STATE_COMPLETE',index,flush=True)
        try:features=feature_diagnostics(manifest,root,budget)
        except Exception as exc:features=dict(status='OMITTED_OR_FAILED',reason=repr(exc),does_not_change_primary_acceptance=True)
        save(root/'feature_diagnostics.json',features)
    except BaseException as exc:
        error=exc;save(root/'failure.json',dict(reason=repr(exc),traceback=traceback.format_exc()))
    finally:
        budget.finish(error)
    source_checks=[c for state in states for row in state if row['is_source_trace'] for c in row['comparisons']]
    candidate_checks=[c for state in states for row in state if not row['is_source_trace'] for c in row['comparisons']]
    pixels=[p for c in source_checks for p in c['pixels']]
    complete=len(budget.branches)==42 and len(states)==2 and all(row['complete'] for state in states for row in state)
    passed=complete and error is None and not budget.stopped and len(pixels)==96 and all(p['bitwise_equal'] for p in pixels)
    passed=passed and all(c['physics']['passed'] and len(c['pixels'])==16 and all(p['bitwise_equal'] for p in c['pixels']) for c in source_checks+candidate_checks)
    report=dict(status='PASS' if passed else 'FAIL',strict_acceptance_unchanged=True,attempted_branches=len(budget.branches),
        source_rgb_matches=sum(p['bitwise_equal'] for p in pixels),source_rgb_images=len(pixels),
        source_physical_checks_pass=bool(source_checks) and all(c['physics']['passed'] for c in source_checks),
        candidate_repeat_checks=len(candidate_checks),candidate_repeat_agreement=bool(candidate_checks) and all(c['physics']['passed'] and all(p['bitwise_equal'] for p in c['pixels']) for c in candidate_checks),
        source_replays_after_candidates=sum(row['source_replay_after_candidate_branches'] for state in states for row in state),
        source_pixel_error_mae_summary=None if not pixels else dict(mean=float(np.mean([p['mae'] for p in pixels])),maximum=max(p['mae'] for p in pixels)),
        per_branch_evidence='state_XX/*/restoration_comparison.json',secondary_features_change_acceptance=False,
        phase2_authorized=False,stop='Completed bounded amended recheck; no retry or automatic extension.')
    save(root/'result.json',report);print(json.dumps(report,indent=2),flush=True)
    if error is not None:raise error


if __name__=='__main__':main()
