"""Per-state V4 execution, only inside the approved audit owner."""
import copy
import hashlib
import json
import pickle
from pathlib import Path
import zlib
import numpy as np
import torch
from PIL import Image
from lewm import decision_headroom_v4_development as v4
from lewm import decision_headroom_v41_development as v41
from lewm.decision_headroom_reference_development import ReferenceGeometry, fixed_target_world
from lewm_genesis.lewm_contract import apply_safety_limits_single
from scripts import run_go2_decision_headroom_branches_development as physics
from scripts.run_go2_decision_headroom_rgb_recheck_development import pixel_comparison
from scripts.qualify_go2_source_decision_packet_development import qualify
from scripts.qualify_go2_articulated_clearance_development import ClearanceQualification


def heads_for(model,config):
    binding=config['execution_input_bindings']['readout_maze']
    head=copy.deepcopy(model.readout)
    head.load_state_dict(torch.load(binding['path'],map_location='cpu',weights_only=False)['model_state_dict'])
    return dict(old_data=model.readout,maze_data=head.eval().requires_grad_(False))


def reactive_tape(packet,selection,limits):
    requested=np.zeros((8,3));requested[:3]=packet['committed_prefix']
    length=selection['command_duration_ns']//100_000_000
    requested[3:3+length]=selection['requested_command']
    last=packet['native_context'][-1]['sensor_state']['control']['applied_command']['values'][-1]
    applied=np.asarray(apply_safety_limits_single(requested.tolist(),last,limits)[0])
    matching=[i for i,tape in enumerate(packet['candidate_applied_commands']) if np.allclose(tape,applied,atol=1e-7,rtol=0)]
    return requested,applied,matching[0] if matching else None


def run(session,source_root,budget,*,model,case):
    config=json.loads(Path('docs/go2_decision_headroom_protocol_v41_2026-09-23.json').read_text())
    session.physics_clock_callback=None
    for hook in list(model._forward_pre_hooks):
        del model._forward_pre_hooks[hook];model._forward_pre_hooks_with_kwargs.pop(hook,None)
    spec=json.loads((source_root/'specification.json').read_text());states=json.loads((source_root/'snapshots.json').read_text())
    heads=heads_for(model,config);articulated=v4.ArticulatedSteps(spec['geometry']['wall_boxes'])
    try:geometry_binding=ClearanceQualification(session,spec,tolerance_m=1e-5).binding
    except Exception as exc:geometry_binding=dict(status='unresolved',reason=repr(exc))
    physics.save(source_root/'native_geometry_binding.json',geometry_binding)
    walls=[dict(center=w['centre_xyz'][:2],size=w['size_xyz'][:2],yaw=w['yaw_rad']) for w in spec['geometry']['wall_boxes']]
    geometry=ReferenceGeometry(walls,[[-2.1,-2.1],[3.4,3.4]],spec['geometry']['spawn_se2_world'][:2],radius_m=.46,clearance_m=.005,resolution_m=.02)
    with np.load(source_root/'native/physics_trace.npz',allow_pickle=False) as arrays:truth={k:arrays[k].copy() for k in physics.TRACE_FIELDS}
    requests=json.loads((source_root/'requests.json').read_text());plans=json.loads((source_root/'planning.json').read_text())
    first=min((s['frame'] for s in states),default=None)
    for state in sorted(states,key=lambda s:s['frame']):
        root=source_root/f"state_{state['frame']:04d}";budget.snapshots.add((case,state['frame']))
        if state['status']!='captured':continue
        try:
            data=(root/'snapshot_packet.pkl.zlib').read_bytes()
            if hashlib.sha256(data).hexdigest()!=state['bundle']['sha256']:raise ValueError('snapshot digest mismatch')
            bundle=pickle.loads(zlib.decompress(data));packet,physical=bundle['decision'],bundle['physical'];stamp=state['measured_ns']
            plan=next(p for p in plans if p['frame']==state['frame'] and 'selection' in p)
            original_request=next(p for p in requests if p['simulator_ns']==stamp)
            fidelity=qualify(packet,physical,model,state['source_controller'],plan,original_request,config['source_decision_tolerances'])
            physics.save(root/'source_input_fidelity.json',fidelity)
            source_valid=fidelity['status']=='PASS'
            recorded=[p for p in requests if stamp<=p['simulator_ns']<stamp+800_000_000]
            source_complete=[p['simulator_ns'] for p in recorded]==list(range(stamp,stamp+800_000_000,20_000_000))
            branch_data={};comparisons=[];repeat=3 if state['frame']==first else 1
            reactive=v4.selector(packet,v4.command_motion(packet),reactive=True) if source_valid else None
            reactive_requested=reactive_applied=bank_index=None
            if reactive is not None:reactive_requested,reactive_applied,bank_index=reactive_tape(packet,reactive,model.limits)
            schedule=[('source_trace_0',None,0)] if source_complete else []
            schedule += [(f'{a}_{j}',i,j) for i,a in enumerate(v4.ACTIONS) for j in range(repeat)]
            if reactive is not None and bank_index is None:schedule += [(f'reactive_{j}',6,j) for j in range(repeat)]
            if source_complete:schedule += [('source_trace_1',None,1),('source_trace_2',None,2)]
            for name,index,j in schedule:
                if index is None:tape=expected=np.array([p['applied_command'] for p in recorded])
                elif index==6:tape=np.repeat(reactive_requested,5,axis=0);expected=np.repeat(reactive_applied,5,axis=0)
                else:tape=np.repeat(packet['candidate_requested_commands'][index],5,axis=0);expected=np.repeat(packet['candidate_applied_commands'][index],5,axis=0)
                actual,terminal=physics.execute(session,physical,tape,root/name,budget,identity=f'{source_root.name}/{root.name}/{name}',geometry=geometry,expected_applied=expected)
                if index is None:
                    compare=physics.compare_trace(actual,truth,stamp,config['restoration_tolerances']);pixels=[]
                    for h in range(1,9):
                        for camera,prefix in [('primary','rgb'),('auxiliary','auxiliary_rgb')]:
                            target=source_root/'native'/f'{prefix}_{state["frame"]+h:04d}.png';replay=root/name/f'{camera}_{100*h:03d}ms.png'
                            pixels.append(pixel_comparison(target,replay) if target.exists() and replay.exists() else dict(bitwise_equal=False,reason='MISSING_FRAME'))
                    comparisons.append(dict(kind='source',name=name,physics=compare,rgb=pixels,terminal=terminal))
                elif j==0:branch_data[index]=(actual,terminal,root/name)
                else:
                    old,old_terminal,oldroot=branch_data[index];compare=physics.compare_trace(actual,old,stamp,config['restoration_tolerances'])
                    frame_equal=all((oldroot/f'{camera}_{h:03d}ms.png').exists() and (root/name/f'{camera}_{h:03d}ms.png').exists() and pixel_comparison(oldroot/f'{camera}_{h:03d}ms.png',root/name/f'{camera}_{h:03d}ms.png')['bitwise_equal'] for h in range(100,801,100) for camera in ('primary','auxiliary'))
                    comparisons.append(dict(kind='candidate_repeat',name=name,physics=compare,rgb_equal=frame_equal,same_terminal=terminal==old_terminal))
            physics.save(root/'restoration.json',dict(source_trace_complete=source_complete,comparisons=comparisons))
            physics_valid=source_complete and all(x['physics']['passed'] for x in comparisons)
            rgb_valid=physics_valid and all(all(p['bitwise_equal'] for p in x['rgb']) if x['kind']=='source' else x['rgb_equal'] for x in comparisons)
            safety=[];traces=[]
            for index in range(6):
                trace,terminal,branch=branch_data[index];traces.append(trace)
                value=articulated.evaluate(trace)
                if not physics_valid or (geometry_binding.get('status')=='unresolved' and not value['contact']):value.update(hard='unresolved',operating='unresolved')
                # Preserve contact even where restoration or clearance is unresolved.
                physics.save(branch/'articulated_v4.json',value);safety.append({k:v for k,v in value.items() if not k.startswith('per_')})
            try:true=v4.true_motion(traces,stamp)
            except ValueError:true=None
            future_rgb=[[np.asarray(Image.open(branch_data[i][2]/f'primary_{h:03d}ms.png').convert('RGB')) for h in range(100,801,100)] for i in range(6)] if rgb_valid and all((branch_data[i][2]/f'primary_{h:03d}ms.png').is_file() for i in range(6) for h in range(100,801,100)) else None
            if source_valid:
                motions=v4.feature_motions(model,heads,packet,future_rgb)
                panel=v4.row_panel(packet,motions,true,state_id=f'{case}/{state["frame"]}')
            else:panel={};motions=None
            filters={}
            for key in ['R2','R5c','R4/old_data','R4/maze_data']:
                if 'eligibility' in panel.get(key,{}):filters[key]={criterion:v4.filter_observation(panel[key]['eligibility'],safety,criterion) for criterion in ('hard','operating')}
            localisation=v41.localisation(packet,truth['base_pose_world'][749],branch_data[0][0]['base_pose_world'][0])
            secondary_costs=[];secondary_reference=dict(status='unresolved',reason='PRIMARY_APPLICABILITY_UNRESOLVED')
            endpoint_strata=[('inside_046m_inflation' if float(geometry.footprint_clearance(t['base_pose_world'][-1,:2]))<0 else 'outside_inflation_below_5mm' if float(geometry.footprint_clearance(t['base_pose_world'][-1,:2]))<.005 else 'reference_free') for t in traces]
            objective=v4.active_objective(packet);costs=[];reference=dict(status='unresolved',reason='VIEW_SEEKING_NO_POSITIONAL_COST')
            if objective['reference_regret_applicable'] and source_valid and physics_valid:
                anchor=truth['base_pose_world'][749];target=fixed_target_world(packet,anchor)
                if target['valid']:
                    refgeo=ReferenceGeometry(walls,[[-2.1,-2.1],[3.4,3.4]],target['target_xy_world'],radius_m=.46,clearance_m=.005,resolution_m=.02)
                    costs=[v4.scalar_reference(refgeo,t,arrival_settling=objective['mode']=='positional_terminal',parameters=config['reference_cost']) if len(t['timestamp_s'])==401 else dict(cost_s=None,status='unresolved',reason='INCOMPLETE_TRACE') for t in traces]
                    reference=v4.reference_panel(costs,safety)
                    secondary_geometry=v41.SecondaryReference(refgeo)
                    secondary_costs=[v4.scalar_reference(secondary_geometry,t,arrival_settling=objective['mode']=='positional_terminal',parameters=config['reference_cost']) if len(t['timestamp_s'])==401 and safety[i]['hard']=='safe' else dict(cost_s=None,status='unresolved',reason='INCOMPLETE_OR_NOT_PHYSICALLY_SAFE') for i,t in enumerate(traces)]
                    secondary_reference=v4.reference_panel(secondary_costs,safety)
            repeat_cost_ranges=[];secondary_repeat_cost_ranges=[]
            if repeat==3 and (reference['status']=='available' or secondary_reference['status']=='available'):
                for i,action in enumerate(v4.ACTIONS):
                    primary_values=[];secondary_values=[]
                    for j in range(3):
                        with np.load(root/f'{action}_{j}'/'physics_trace.npz',allow_pickle=False) as a:repeated={k:a[k].copy() for k in physics.TRACE_FIELDS}
                        if len(repeated['timestamp_s'])!=401:continue
                        for geo,values in ((refgeo,primary_values),(secondary_geometry,secondary_values)):
                            rc=v4.scalar_reference(geo,repeated,arrival_settling=objective['mode']=='positional_terminal',parameters=config['reference_cost'])
                            if rc['cost_s'] is not None:values.append(rc['cost_s'])
                    for values,result,ref in ((primary_values,repeat_cost_ranges,reference),(secondary_values,secondary_repeat_cost_ranges,secondary_reference)):
                        spread=max(values)-min(values) if len(values)==3 else None
                        result.append(dict(action=action,range_s=spread))
                        if i in ref.get('physical',[]) and (spread is None or spread>.025):ref.update(status='unresolved',reason='SPOT_CHECK_REFERENCE_REPEAT_VARIABILITY')
            for name,row in panel.items():
                i=row.get('action_index',(row.get('selection') or {}).get('action_index'))
                if name=='R5r':i=bank_index
                if i is not None and 0<=i<6:row['physical_status']=safety[i]
            if secondary_reference['status']=='available':
                for name,row in panel.items():
                    i=bank_index if name=='R5r' else row.get('action_index',(row.get('selection') or {}).get('action_index'))
                    row['secondary_regret_s']=secondary_costs[i]['cost_s']-secondary_reference['cost_s'] if i in secondary_reference['margin'] else None
            if reference['status']=='available':
                panel['R1']=dict(status='available',action_index=reference['action_index'])
                for row in panel.values():
                    i=row.get('action_index',(row.get('selection') or {}).get('action_index'))
                    if i is None:continue
                    if row is panel.get('R5r') and bank_index is None:continue
                    if row is panel.get('R5r'):i=bank_index
                    row['reference_practical_near_optimal']=i in reference['practical_near_optimal_indices']
                    row['unnecessary_hold']=i==0 and 0 in reference['margin'] and costs[0]['cost_s']-reference['cost_s']>.10
                    row['physical_status']=safety[i];row['regret_s']=costs[i]['cost_s']-reference['cost_s'] if i in reference['margin'] else None
                    if 'eligibility' in row:
                        acceptable=[j for j in reference['margin'] if row['eligibility']['candidates'][j]['eligible']]
                        row['excess_rejection_cost_s']=min(costs[j]['cost_s'] for j in acceptable)-reference['cost_s'] if acceptable else None
            if reactive is not None and bank_index is None:
                t,terminal,branch=branch_data[6];safe=articulated.evaluate(t);physics.save(branch/'articulated_v4.json',safe)
                offcost=v4.scalar_reference(refgeo,t,arrival_settling=objective['mode']=='positional_terminal',parameters=config['reference_cost']) if reference['status']=='available' and len(t['timestamp_s'])==401 else dict(cost_s=None)
                panel.get('R5r',{}).update(off_bank=True,physical_status={k:v for k,v in safe.items() if not k.startswith('per_')},signed_bank_gap_s=offcost['cost_s']-reference['cost_s'] if offcost['cost_s'] is not None and safe['operating']=='safe' else None)
            physics.save(root/'audit_v4.json',dict(objective=objective,sampling=state,source_input_valid=source_valid,physics_valid=physics_valid,rgb_valid=rgb_valid,safety=safety,reference=reference,costs=costs,repeat_cost_ranges=repeat_cost_ranges,rows=panel,filter_audit=filters,localisation=localisation,endpoint_clearance_strata=endpoint_strata,secondary_reference=secondary_reference,secondary_costs=secondary_costs,secondary_repeat_cost_ranges=secondary_repeat_cost_ranges,phase2=True,protocol_version='4.1'))
            budget.check('state_complete',force=True)
        except Exception as exc:
            physics.save(root/'state_unresolved.json',dict(reason=repr(exc),retry=False,scope='This state only; retained partial evidence remains'))
            if budget.stopped:raise
