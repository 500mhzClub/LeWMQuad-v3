"""Retained-input provider for the complete V4.2 owner, and output validation.

Only the six qualified states are admitted. No alternate scoring implementation;
the normal branch owner, frozen models, row adapters and reader are called.
"""
import hashlib
import json
from pathlib import Path
import shutil
from types import SimpleNamespace
import numpy as np
from PIL import Image
from lewm import decision_headroom_v4_development as v4
from lewm import decision_headroom_json_v42_development as output_json
from scripts import run_go2_decision_headroom_branches_development as physics


def write(path,value):
    with path.open('x') as f:json.dump(value,f,indent=2)


def copy_record(source,target):
    if source.suffix=='.json':write(target,json.loads(source.read_text()))
    else:
        shutil.copyfile(source,target)
        if hashlib.sha256(source.read_bytes()).digest()!=hashlib.sha256(target.read_bytes()).digest():
            raise output_json.OutputFailure('retained file readback mismatch: '+str(target))


class RetainedSources:
    def __init__(self,root,budget,config):
        self.root,self.budget,self.config=root,budget,config
        c=json.loads(Path('docs/go2_decision_headroom_remaining_phase1_approved_v3_2026-09-23.json').read_text())
        prior=Path(c['execution_caps']['output_root']);corrected=prior.parent/'go2_decision_headroom_rgb_restore_recheck_v1_attempt_001'
        self.states={}
        for i,s in enumerate(c['previous_corrected_states']):self.states[i]=(Path(s['source_root']),s,corrected/f'state_{i:02d}')
        for i,case in enumerate((2,6,7,8)):
            source=prior/f'source_{i:02d}';state=json.loads((source/'snapshots.json').read_text())[0]
            self.states[case]=(source,state,source/f'state_{state["frame"]:04d}')
        write(root/'implementation_check_scope.json',dict(implementation_only=True,science_panel=False,
            six_state_decision_sha256=[s['decision']['sha256'] for _,s,_ in self.states.values()],
            retained_traces=True,new_physics_steps=0,caps=config['execution_caps'],
            note='Same complete audit owner; retained source/branch provider replaces physics only. No source missions re-executed.'))
        self.model=None

    def run(self,case):
        from scripts import run_go2_headroom_v42_branches_development as branches
        source,state,retained=self.states[case];budget=self.budget
        budget.start_source(case)
        target=self.root/f'source_{case:02d}';target.mkdir();(target/'native').mkdir()
        destination=target/f'state_{state["frame"]:04d}';destination.mkdir()
        bound=source/f'state_{state["frame"]:04d}'
        packet=physics.load_bound(bound/'decision.pkl',state['decision'])
        physical=physics.load_bound(bound/'physical.pkl',state['physical'])
        data,binding=v4.encode_snapshot(dict(decision=packet,physical=physical))
        budget.admit_write(len(data));(destination/'snapshot_packet.pkl.zlib').write_bytes(data)
        objective=v4.active_objective(packet)
        metadata=dict(state,status='captured',bundle=binding,weight=1.,phase=objective['phase'],implementation_only=True)
        write(destination/'snapshot.json',metadata);write(target/'snapshots.json',[metadata])
        for name in ('specification.json','requests.json','planning.json','acquisitions.json','poses.json','mission.json','pipeline_faults.json','model_identity.json','actuator_identity.json','dense_model_calls.json','result.json'):
            if (source/name).exists():copy_record(source/name,target/name)
        copy_record(source/'native/physics_trace.npz',target/'native/physics_trace.npz')
        for name in ('in_memory_camera_observations.json','camera_terminal_identity.json','contact_events.json','physics_topology.json'):
            if (source/'native'/name).exists():copy_record(source/'native'/name,target/'native'/name)
        for frame in range(state['frame']-10,state['frame']+9):
            for prefix in ('rgb','auxiliary_rgb'):
                name=f'{prefix}_{frame:04d}.png'
                if (source/'native'/name).is_file():copy_record(source/'native'/name,target/'native'/name)
        if self.model is None:self.model=v4.deployed.load_dense_navigation_model('action',readout_arm='maze_view_old_data')
        identity=source/'collision_geometry_identity.json'
        geometry_binding=json.loads(identity.read_text()) if identity.is_file() else dict(status='unresolved',reason='NO_RETAINED_NATIVE_GEOMETRY_BINDING_FOR_THIS_SOURCE')
        geometry_binding=dict(geometry_binding,retained_evidence_only=True,source_binding=str(identity))
        old_execute,old_geometry=physics.execute,branches.ClearanceQualification
        def execute(session,snapshot,requested,output,meter,*,identity,geometry,expected_applied):
            old=retained/output.name
            if not old.is_dir():
                raise RuntimeError('RETAINED_ASSIGNMENT_UNAVAILABLE: '+str(old))
            commands=json.loads((old/'commands.json').read_text())
            np.testing.assert_allclose(commands['requested'],requested,atol=1e-7,rtol=0)
            np.testing.assert_allclose(commands['applied'],expected_applied,atol=1e-7,rtol=0)
            output.mkdir()
            for name in ('physics_trace.npz','contact_events.json','commands.json','frames.json','swept_clearance.json','result.json'):
                copy_record(old/name,output/name)
            for h in range(100,801,100):
                for camera in ('primary','auxiliary'):
                    name=f'{camera}_{h:03d}ms.png'
                    if (old/name).is_file():copy_record(old/name,output/name)
            meter.check('retained_branch_readback')
            with np.load(output/'physics_trace.npz',allow_pickle=False) as a:trace={k:a[k].copy() for k in physics.TRACE_FIELDS}
            return trace,json.loads((output/'result.json').read_text())['terminal']
        try:
            physics.execute=execute
            branches.ClearanceQualification=lambda *a,**k:SimpleNamespace(binding=geometry_binding)
            branches.run(SimpleNamespace(physics_clock_callback=None),target,budget,model=self.model,case=case)
            failure=destination/'state_unresolved.json'
            if failure.exists():raise RuntimeError('PRELAUNCH_NON_CONVERTER_DEFECT: '+failure.read_text())
            audit=destination/'audit_v4.json'
            if not audit.is_file():raise RuntimeError('missing complete per-state audit output')
            write(target/'retained_check_result.json',dict(implementation_only=True,source=str(source),branches=str(retained),new_physics_steps=0))
        finally:
            physics.execute,branches.ClearanceQualification=old_execute,old_geometry
            budget.finish_source(case,None)


def assemble_outputs(root,config,*,implementation_only):
    """Versioned assembly and memo inputs from written outputs, no new scoring."""
    states=[]
    for case in range(24):
        source=root/f'source_{case:02d}';path=source/'snapshots.json'
        if not path.exists():continue
        for state in json.loads(path.read_text()):
            audit=source/f'state_{state["frame"]:04d}'/'audit_v4.json'
            states.append(dict(case=case,frame=state['frame'],path=str(audit.relative_to(root)),
                audit=json.loads(audit.read_text()) if audit.exists() else None,
                missing_reason=None if audit.exists() else 'PER_STATE_OUTPUT_UNAVAILABLE'))
    write(root/'branch_panel_v42.json',dict(schema='headroom_branch_panel.v4.2',implementation_only=implementation_only,states=states))
    analysis=json.loads((root/'analysis_v42.json').read_text())
    write(root/'memo_inputs_v42.json',dict(schema='headroom_memo_inputs.v4.2',implementation_only=implementation_only,
        primary_quantities=analysis['primary_quantities'],layout_clustered=analysis['layout_clustered'],
        cell_coverage=analysis['cell_coverage'],rules=config['memo_rules'],inconclusive_regret_supports_neither_stopping_nor_continuing=True))


def validate_outputs(root):
    results=[]
    for directory,dirs,files in __import__('os').walk(root):
        dirs[:]=[d for d in dirs if d!='sealed' and not d.startswith('sealed_')]
        for name in files:
            if name=='sealed_test.json':continue
            p=Path(directory)/name
            if p.suffix=='.json':output_json.schema(p,json.loads(p.read_text()))
            elif p.suffix=='.jsonl':
                for line in p.read_text().splitlines():output_json.schema(p,json.loads(line))
            elif p.suffix=='.png':
                with Image.open(p) as im:assert im.mode=='RGB' and im.size==(640,480);im.load()
            elif p.suffix=='.npz':
                with np.load(p,allow_pickle=False) as a:
                    assert all(k in a for k in physics.TRACE_FIELDS)
                    n=len(a['timestamp_s']);assert all(len(a[k])==n for k in physics.TRACE_FIELDS)
            elif p.name=='snapshot_packet.pkl.zlib':
                import pickle,zlib
                value=pickle.loads(zlib.decompress(p.read_bytes()));assert set(value)=={'physical','decision'}
            elif p.suffix=='.md':assert p.read_text()
            else:raise RuntimeError('unvalidated output file type: '+str(p))
            results.append(dict(path=str(p.relative_to(root)),schema_validated=True,sha256=hashlib.sha256(p.read_bytes()).hexdigest()))
    write(root/'output_validation.json',dict(status='PASS',implementation_only=True,files=results,new_physics_steps=0))
