"""User-approved four-cell qualification, stopping before Phase 2."""
import copy
import hashlib
import json
import os
from pathlib import Path
import shutil
import threading
import time
import traceback

import numpy as np
import psutil
import torch

from lewm.eligible_floor_registration_development import bind
from lewm.decision_headroom_reference_development import ReferenceGeometry
from scripts import run_go2_decision_headroom_source_development as collector
from scripts import run_go2_decision_headroom_branches_development as branches
from scripts import time_go2_decision_headroom_components_development as timing
from scripts.run_go2_decision_headroom_pilot_development import PilotBudget, REPO, save
from scripts.run_go2_decision_headroom_rgb_recheck_development import pixel_comparison
from scripts.qualify_go2_source_decision_packet_development import qualify
from scripts.qualify_go2_articulated_clearance_development import ClearanceQualification

CONFIG=Path('docs/go2_decision_headroom_remaining_phase1_approved_v3_2026-09-23.json')
ACTIVE_MODEL=None
SOURCE_LOAD=collector.source.load_dense_navigation_model


def load_model(*args,**kwargs):
    global ACTIVE_MODEL
    ACTIVE_MODEL=SOURCE_LOAD(*args,**kwargs)
    return ACTIVE_MODEL


class Budget(PilotBudget):
    def reserve_branch(self, identity):
        if len(self.branches)>=len(self.schedule) or identity!=self.schedule[len(self.branches)]:
            raise ValueError('fixed branch order/attempt cap differs')
        self.check('branch_admission',force=True)
        self.branches.add(identity);self.event('branch_reserved',identity=identity,physics_ns=800_000_000)


def source_fidelity(source,state,model,config,destination):
    root=source/f"state_{state['frame']:04d}"
    packet=branches.load_bound(root/'decision.pkl',state['decision'])
    physical=branches.load_bound(root/'physical.pkl',state['physical'])
    plans=json.loads((source/'planning.json').read_text())
    plan=next(p for p in plans if p['frame']==state['frame'] and 'selection' in p)
    request=next(r for r in json.loads((source/'requests.json').read_text()) if r['simulator_ns']==state['measured_ns'])
    report=qualify(packet,physical,model,state['source_controller'],plan,request,config['source_decision_tolerances'])
    report['input_sha256']={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in (root/'snapshot.json',root/'decision.pkl',root/'physical.pkl',source/'planning.json',source/'requests.json')}
    save(destination,report)
    if report['status']!='PASS':raise RuntimeError('SOURCE_DECISION_REPRODUCTION_FAILED: '+str(destination))
    return packet,physical


def run_branches(session, source_root, budget, *, tolerances):
    config=json.loads(CONFIG.read_text())
    session.physics_clock_callback=None
    # The source has finished; detach its observational hook before replaying.
    for hook in list(ACTIVE_MODEL._forward_pre_hooks):
        del ACTIVE_MODEL._forward_pre_hooks[hook]
        ACTIVE_MODEL._forward_pre_hooks_with_kwargs.pop(hook,None)
    states=json.loads((source_root/'snapshots.json').read_text())
    if len(states)!=1 or states[0]['frame']!=132:
        raise RuntimeError('FIXED_SNAPSHOT_UNAVAILABLE; no replacement')
    state=states[0];stamp=state['measured_ns'];root=source_root/'state_0132'
    decision,snapshot=source_fidelity(source_root,state,ACTIVE_MODEL,config,root/'source_input_qualification.json')
    budget.check('source_decision_reproduced',force=True)
    spec=json.loads((source_root/'specification.json').read_text())
    clearance=ClearanceQualification(session,spec,tolerance_m=config['native_geometry_identity_tolerance'])
    save(source_root/'collision_geometry_identity.json',clearance.binding)
    walls=[dict(center=w['centre_xyz'][:2],size=w['size_xyz'][:2],yaw=w['yaw_rad']) for w in spec['geometry']['wall_boxes']]
    extents=np.array([w['center'] for w in walls])
    geometry=ReferenceGeometry(walls,[extents.min(axis=0)-2,extents.max(axis=0)+2],spec['geometry']['spawn_se2_world'][:2],radius_m=.46,clearance_m=.005,resolution_m=.02)
    with np.load(source_root/'native/physics_trace.npz',allow_pickle=False) as a:
        truth={k:a[k].copy() for k in branches.TRACE_FIELDS}
    stamps=np.rint(truth['timestamp_s']*1e9).astype(np.int64)
    mask=(stamps>=stamp)&(stamps<=stamp+800_000_000)
    source_contact=json.loads((source_root/'native/contact_events.json').read_text())
    save(root/'source_articulated_clearance.json',clearance.trace({k:v[mask] for k,v in truth.items()},source_contact,budget=budget))
    requests=json.loads((source_root/'requests.json').read_text())
    trace=[r for r in requests if stamp<=r['simulator_ns']<stamp+800_000_000]
    if [r['simulator_ns'] for r in trace]!=list(range(stamp,stamp+800_000_000,20_000_000)):
        raise RuntimeError('SOURCE_TRACE_INCOMPLETE')
    source_commands=np.array([r['applied_command'] for r in trace])
    cameras={r['measured_ns']:r for r in json.loads((source_root/'native/in_memory_camera_observations.json').read_text())['frames']}
    previous={};results=[]
    for name in config['remaining_phase1']['branch_order_per_state']:
        is_source=name.startswith('source_trace_')
        if is_source:tape=projected=source_commands
        else:
            action,repeat=name.rsplit('_',1);index=branches.ACTIONS.index(action)
            tape=np.repeat(decision['candidate_requested_commands'][index],5,axis=0)
            projected=np.repeat(decision['candidate_applied_commands'][index],5,axis=0)
        output=root/name
        actual,terminal=branches.execute(session,snapshot,tape,output,budget,
            identity=f'{source_root.name}/state_0132/{name}',geometry=geometry,expected_applied=projected)
        frames=json.loads((output/'frames.json').read_text());comparisons=[]
        if is_source:
            pixels=[]
            for f in frames:
                original=cameras[stamp+f['offset_ms']*1_000_000]
                for label,prefix in [('primary','rgb'),('auxiliary','auxiliary_rgb')]:
                    path=source_root/'native'/f"{prefix}_{original['frame']:04d}.png"
                    pixels.append(dict(offset_ms=f['offset_ms'],camera=label,**pixel_comparison(path,output/f['images'][label]['path'])))
            comparisons.append(dict(physics=branches.compare_trace(actual,truth,stamp,tolerances),pixels=pixels))
        else:
            for prior,prior_terminal,prior_output in previous.get(action,[]):
                pixels=[dict(offset_ms=f['offset_ms'],camera=label,**pixel_comparison(prior_output/f['images'][label]['path'],output/f['images'][label]['path'])) for f in frames for label in ('primary','auxiliary')]
                comparisons.append(dict(physics=branches.compare_trace(actual,prior,stamp,tolerances),pixels=pixels,same_terminal=terminal==prior_terminal))
            previous.setdefault(action,[]).append((actual,terminal,output))
        passed=terminal is None and len(frames)==8 and all(c['physics']['passed'] and len(c['pixels'])==16 and all(p['bitwise_equal'] for p in c['pixels']) for c in comparisons)
        row=dict(name=name,is_source=is_source,passed=passed,terminal=terminal,comparisons=comparisons)
        save(output/'restoration_comparison.json',row);results.append(row)
        if not passed:raise RuntimeError('STRICT_RESTORATION_OR_REPEAT_FAILURE: '+str(output))
        contacts=json.loads((output/'contact_events.json').read_text())
        save(output/'articulated_clearance.json',clearance.trace(actual,contacts,budget=budget))
        print('QUALIFICATION_BRANCH',source_root.name,name,flush=True)
    save(source_root/'qualification_result.json',dict(status='RESTORATION_AND_SOURCE_INPUT_PASS_CLEARANCE_UNRESOLVED',branches=len(results),source_replay_rgb_matches=48,source_input=True,articulated_swept_clearance=False,phase2_authorized=False))


def main():
    global ACTIVE_MODEL
    config=json.loads(CONFIG.read_text());caps=config['execution_caps'];root=Path(caps['output_root'])
    if root.exists():raise RuntimeError('No restart or replacement attempt')
    for path,sha in config['frozen_source_sha256'].items():
        if hashlib.sha256(Path(path).read_bytes()).hexdigest()!=sha:raise RuntimeError('source identity differs: '+path)
    for binding in config['execution_input_bindings'].values():
        p=Path(binding['path'])
        if p.stat().st_size!=binding['bytes'] or hashlib.file_digest(p.open('rb'),'sha256').hexdigest()!=binding['sha256']:
            raise RuntimeError('checkpoint/input identity differs: '+str(p))
    collector.require_stage_a_closed(root)
    os.sched_setaffinity(0,caps['compute_caps']['cpu_affinity']);torch.set_num_threads(4)
    root.mkdir();(root/'scratch').mkdir()
    admission=dict(owner_pid=os.getpid(),owner_created=psutil.Process().create_time(),caps_sha256=hashlib.sha256(CONFIG.read_bytes()).hexdigest(),restoration_tolerances=config['restoration_tolerances'],
        cache_paths=[str(Path.home()/'.cache'/s) for s in ('genesis','quadrants','gstaichi','triton','torch','mesa_shader_cache','mesa_shader_cache_db')]+[str(REPO/'.generated/box_meshes'),'/tmp/torchinductor_'+Path.home().name])
    save(root/'pilot_execution_admission.json',admission)
    budget=Budget(root,caps,admission)
    budget.schedule=[f'source_{i:02d}/state_0132/{name}' for i in range(4) for name in config['remaining_phase1']['branch_order_per_state']]
    error=None
    try:
        budget.check('initial_admission',force=True)
        ACTIVE_MODEL=SOURCE_LOAD('action',readout_arm='maze_view_old_data')
        for index,state in enumerate(config['previous_corrected_states']):
            source_fidelity(Path(state['source_root']),state,ACTIVE_MODEL,config,root/f'previous_state_{index}_source_qualification.json')
            budget.check('previous_source_packet_qualified',force=True)
        del ACTIVE_MODEL;ACTIVE_MODEL=None
        torch.cuda.empty_cache()
        collector.source.load_dense_navigation_model=load_model
        branches.run=run_branches
        timing.run=lambda *args,**kwargs: None
        # Audit globals only; frozen controller/model/renderer functions stay unchanged.
        run=bind(collector.run,CAPS=CONFIG,ASSIGNMENTS=tuple((s['layout'],s['controller']) for s in config['remaining_phase1']['source_assignments']))
        # The old collector consumes the legacy caps shape. Supply it without
        # changing its source, via a frozen audit-local read adapter.
        class CapsPath:
            def read_text(self):return json.dumps(caps)
            def read_bytes(self):return CONFIG.read_bytes()
        run=bind(run,CAPS=CapsPath())
        for case in range(4):
            run(case)
            ACTIVE_MODEL=None;torch.cuda.empty_cache()
        save(root/'result.json',dict(status='BOUNDED_BATCH_COMPLETE_WITH_CLEARANCE_BLOCKER',source_cells=4,branches=84,phase2_authorized=False))
    except BaseException as exc:
        error=exc
        save(root/'failure.json',dict(reason=repr(exc),traceback=traceback.format_exc(),retry_authorized=False))
        save(root/'result.json',dict(status='STOPPED_WITH_FINITE_BLOCKER',reason=repr(exc),source_attempts=len(budget.sources),branches=len(budget.branches),phase2_authorized=False))
    finally:
        budget.finish(error)
    print(json.dumps(json.loads((root/'result.json').read_text())),flush=True)
    if error:raise error


if __name__=='__main__':main()
