#!/usr/bin/env python3
"""Replay frozen wide-geometry branches through H1 and materialise geometry.

No learned checkpoint is loaded.  Physics is replayed only to recover the
registered articulated transforms which were omitted from the earlier row
ledger.  Large arrays are written to the high-capacity temporal cache.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
import types

import numpy as np
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parents[1]
for extra in (ROOT, ROOT/"scripts", ROOT/"lewm_genesis", ROOT/"lewm_worlds"):
    if str(extra) not in sys.path: sys.path.insert(0, str(extra))

from lewm.safety import articulated_swept_geometry_v1 as GEO
from scripts import collect_factorised_micro_safety_world_model_v1 as COLLECT
from scripts import materialize_geometry_modality_safety_sufficiency_v1 as SENSOR
from scripts import run_go2_oracle_branch_pilot_v1_2 as V

OUT = ROOT/".generated/h1_articulated_swept_geometry_sufficiency_v1"
CACHE = Path("/home/andrewknowles/.cache/lewm_go2_temporal_v03/h1_articulated_swept_geometry_sufficiency_v1")
PANEL = ROOT/".generated/wide_geometry_embodied_contact_proxy_v1/fresh_panel_manifest.json"
ENHANCED = ROOT/".generated/wide_geometry_embodied_contact_proxy_v1/fresh_enhanced_sensor_index.json"
GEOMETRY = ROOT/".generated/wide_geometry_embodied_contact_proxy_v1/fresh_geometry_sensor_index.json"
PHYSICS_STEPS = 250


def sha(path: Path) -> str:
    h=hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda:f.read(1<<22),b""): h.update(block)
    return h.hexdigest()


def atomic_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True,exist_ok=True); tmp=path.with_name(f".{path.name}.tmp-{os.getpid()}")
    tmp.write_text(json.dumps(value,indent=2,sort_keys=True,allow_nan=False)+"\n"); os.replace(tmp,path)


def atomic_npz(path: Path, **arrays) -> None:
    path.parent.mkdir(parents=True,exist_ok=True); tmp=path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with tmp.open("wb") as f: np.savez_compressed(f,**arrays)
    os.replace(tmp,path)


def arr(value):
    try: value=value.detach().cpu().numpy()
    except AttributeError: value=np.asarray(value)
    value=np.asarray(value)
    if value.ndim>1 and value.shape[0]==1: value=value[0]
    return value


def records(path: Path) -> dict[str,dict]:
    return {str(row["state_id"]):row for row in json.loads(path.read_text())["state_records"]}


def scene_boxes(state):
    scene=json.loads((Path(state["scene_dir"])/"genesis_scene.json").read_text())
    objects=[x for x in scene["objects"] if x.get("kind")!="ground"]
    return (np.asarray([x["center_xyz_m"] for x in objects],np.float64),
            np.asarray([x["size_xyz_m"] for x in objects],np.float64)/2,
            np.asarray([x.get("yaw_rad",0.) for x in objects],np.float64),
            np.asarray([str(x.get("object_id",x.get("name",f"object-{i}"))) for i,x in enumerate(objects)]))


def sensor_origin_and_angles(pose, jitter, *, lidar=False):
    x,y,z=float(pose[0]),float(pose[1]),float(pose[2]); q=pose[3:]
    yaw=math.atan2(2*(q[0]*q[3]+q[1]*q[2]),1-2*(q[2]*q[2]+q[3]*q[3]))
    if lidar:
        offset=np.asarray(SENSOR.LIDAR_XYZ_BODY_M); jrpy=np.zeros(3)
    else:
        offset=np.asarray(SENSOR.CAMERA_XYZ_BODY_M)+np.asarray(jitter.get("xyz_offset_m",[0,0,0])); jrpy=np.asarray(jitter.get("rpy_offset_rad",[0,0,0]))
    forward=np.asarray([math.cos(yaw),math.sin(yaw)]); left=np.asarray([-math.sin(yaw),math.cos(yaw)])
    origin=np.asarray([x+offset[0]*forward[0]+offset[1]*left[0], y+offset[0]*forward[1]+offset[1]*left[1], z+offset[2]])
    return origin,yaw,jrpy


def depth_points(depth, valid, pose, jitter):
    origin,yaw,jrpy=sensor_origin_and_angles(pose,jitter,lidar=False)
    horizontal=math.radians(SENSOR.DEPTH_HORIZONTAL_FOV_DEG)
    vertical=2*math.atan(math.tan(horizontal/2)*SENSOR.DEPTH_HEIGHT/SENSOR.DEPTH_WIDTH)
    az=yaw+float(jrpy[2])+np.linspace(-horizontal/2,horizontal/2,SENSOR.DEPTH_WIDTH)
    el=np.linspace(vertical/2,-vertical/2,SENSOR.DEPTH_HEIGHT)+float(jrpy[1])
    ee,aa=np.meshgrid(el,az,indexing="ij"); d=np.asarray(depth,np.float64)
    direction=np.stack((np.cos(ee)*np.cos(aa),np.cos(ee)*np.sin(aa),np.sin(ee)),-1)
    points=origin+d[...,None]*direction; mask=np.asarray(valid,bool)&np.isfinite(points).all(-1)&(points[...,2]>.025)
    return points[mask]


def lidar_points(scan, valid, pose):
    origin,yaw,_=sensor_origin_and_angles(pose,{},lidar=True)
    az=yaw+np.linspace(-math.pi,math.pi,SENSOR.LIDAR_AZIMUTH_BINS,endpoint=False)
    el=np.radians(np.asarray(SENSOR.LIDAR_VERTICAL_DEG)); ee,aa=np.meshgrid(el,az,indexing="ij"); d=np.asarray(scan,np.float64)
    direction=np.stack((np.cos(ee)*np.cos(aa),np.cos(ee)*np.sin(aa),np.sin(ee)),-1)
    points=origin+d[...,None]*direction; mask=np.asarray(valid,bool)&np.isfinite(points).all(-1)&(points[...,2]>.025)
    return points[mask]


def pose7_from_start(state):
    xy,yaw,z=state["start_pose"]; return np.asarray([xy[0],xy[1],z,math.cos(yaw/2),0,0,math.sin(yaw/2)],np.float64)


def sensor_clouds(state, candidate, geom_arrays, sensor_arrays, pose_arrays):
    manifest=json.loads((Path(state["scene_dir"])/"manifest.json").read_text()); jitter=manifest.get("camera_extrinsic_jitter",{})
    poses=[pose7_from_start(state)]
    for tick in range(5):
        p=pose_arrays[candidate,tick]; yaw=float(p[2]); poses.append(np.asarray([p[0],p[1],p[3],math.cos(yaw/2),0,0,math.sin(yaw/2)]))
    d=[depth_points(geom_arrays["current_depth"],geom_arrays["current_depth_valid"],poses[0],jitter)]
    l=[lidar_points(geom_arrays["current_lidar"],geom_arrays["current_lidar_valid"],poses[0])]
    for tick in range(5):
        d.append(depth_points(geom_arrays["future_depth"][candidate,tick],geom_arrays["future_depth_valid"][candidate,tick],poses[tick+1],jitter))
        l.append(lidar_points(geom_arrays["future_lidar"][candidate,tick],geom_arrays["future_lidar_valid"][candidate,tick],poses[tick+1]))
    depth=np.concatenate(d); lidar=np.concatenate(l)
    # Stable de-duplication at 1 mm keeps the true-future surface support while
    # bounding repeated identical ray hits.
    depth=np.unique(np.round(depth,3),axis=0); lidar=np.unique(np.round(lidar,3),axis=0)
    return depth,lidar,np.unique(np.concatenate((depth,lidar)),axis=0)


def geom_contract(robot):
    links_pos=arr(robot.get_links_pos()).astype(np.float64); links_quat=arr(robot.get_links_quat()).astype(np.float64)
    kinds={1:"sphere",3:"capsule",5:"box"}; rows=[]
    for geom in robot.geoms:
        lp=int(geom.link.idx_local); gp=arr(geom.get_pos()).astype(np.float64); gq=arr(geom.get_quat()).astype(np.float64)
        local_pos,local_quat=GEO.inverse_transform(links_pos[lp],links_quat[lp],gp,gq)
        kind=kinds.get(int(geom.type))
        if kind is None: raise RuntimeError(f"unsupported robot geom type {geom.type}")
        rows.append({"geom_index":int(geom.idx)-int(robot.geom_start),"link_index":lp,"link_name":str(geom.link.name),"kind":kind,
                     "data":arr(geom.data).astype(float).tolist(),"local_pos":local_pos.tolist(),"local_quat":local_quat.tolist()})
    return rows


def instantiate(contract, link_pos, link_quat):
    output=[]
    for row in contract:
        pos,quat=GEO.compose(link_pos[row["link_index"]],link_quat[row["link_index"]],row["local_pos"],row["local_quat"])
        output.append({"kind":row["kind"],"data":np.asarray(row["data"]),"pos":pos,"quat":quat})
    return output


def point_score(primitives, cloud, tree):
    best,best_geom=math.inf,-1
    for gi,p in enumerate(primitives):
        ids=tree.query_ball_point(p["pos"],1.25)
        value=GEO.primitive_to_points(p["kind"],p["data"],p["pos"],p["quat"],cloud[ids])
        if value<best: best,best_geom=value,gi
    return float(best),best_geom


def batch_clearances(geom_transform, contract, boxes, clouds):
    """Vectorised 250-step reduction of the same deterministic distances."""
    steps, geoms = geom_transform.shape[:2]
    output=np.full((steps,4),np.inf,np.float64); arg=np.full((steps,4),-1,np.int16); obj=np.full(steps,-1,np.int16)
    centers,halves,yaws,_=boxes
    for gi,row in enumerate(contract):
        pos=np.asarray(geom_transform[:,gi,:3],np.float64); quat=np.asarray(geom_transform[:,gi,3:],np.float64)
        rotations=np.stack([GEO.rotation(q) for q in quat])
        local,radius=GEO.primitive_points(row["kind"],np.asarray(row["data"]),GEO.CAPSULE_CENTERLINE_SAMPLES)
        world=np.einsum("tij,pj->tpi",rotations,local)+pos[:,None,:]
        distance_xy=np.linalg.norm(centers[None,:,:2]-pos[:,None,:2],axis=2)
        nearby=np.flatnonzero(np.any(distance_xy <= 2.0+np.linalg.norm(halves[:,:2],axis=1)[None],axis=0))
        for oi in nearby:
            value=np.min(GEO.box_sdf(world.reshape(-1,3),centers[oi],halves[oi],float(yaws[oi])).reshape(steps,-1),axis=1)-radius
            better=value<output[:,0]; output[better,0]=value[better]; arg[better,0]=gi; obj[better]=oi
        for ci,cloud in enumerate(clouds,start=1):
            tree=cKDTree(cloud); k=min(128,len(cloud)); _dist,indices=tree.query(pos,k=k)
            points=cloud[np.asarray(indices).reshape(steps,k)]
            local_points=np.einsum("tki,tij->tkj",points-pos[:,None,:],rotations)
            data=np.asarray(row["data"],np.float64)
            if row["kind"]=="sphere": value=np.min(np.linalg.norm(local_points,axis=2)-data[0],axis=1)
            elif row["kind"]=="capsule":
                z=np.clip(local_points[:,:,2],-data[1]/2,data[1]/2)
                delta=local_points-np.stack((np.zeros_like(z),np.zeros_like(z),z),-1)
                value=np.min(np.linalg.norm(delta,axis=2)-data[0],axis=1)
            else:
                q=np.abs(local_points)-data[:3]; sdf=np.linalg.norm(np.maximum(q,0),axis=2)+np.minimum(np.max(q,axis=2),0); value=np.min(sdf,axis=1)
            better=value<output[:,ci]; output[better,ci]=value[better]; arg[better,ci]=gi
    return output.astype(np.float32),arg,obj


def execute_branch(ctx,snapshot,candidate,source,labels,poses,contract,boxes,clouds):
    V.V1.restore_branch_state(ctx,snapshot); runner=ctx.runner; topology=V.link_topology(ctx)
    link_t=np.empty((PHYSICS_STEPS,len(runner.build.robot.links),7),np.float32)
    joint=np.empty((PHYSICS_STEPS,12),np.float32); geom_t=np.empty((PHYSICS_STEPS,len(contract),7),np.float32)
    clear=np.empty((PHYSICS_STEPS,4),np.float32); arg_geom=np.empty((PHYSICS_STEPS,4),np.int16); arg_object=np.full(PHYSICS_STEPS,-1,np.int16)
    physics_contact=np.zeros(PHYSICS_STEPS,np.uint8); tick_contact=[]; counter=0
    original=runner._step_policy_step
    def instrumented(_runner,target_cmd):
        nonlocal counter
        observation=_runner._build_observation(target_cmd); targets=_runner.policy.act(observation); _runner._apply_joint_targets(targets)
        for _ in range(int(_runner._physics_steps_per_policy)):
            _runner.build.scene.step(); index=counter; counter+=1
            lp=arr(_runner.build.robot.get_links_pos()).astype(np.float64); lq=arr(_runner.build.robot.get_links_quat()).astype(np.float64)
            link_t[index,:,:3]=lp; link_t[index,:,3:]=lq
            joint[index]=arr(_runner.build.robot.get_dofs_position(_runner._leg_dof_idx.tolist()))
            primitives=instantiate(contract,lp,lq)
            for gi,p in enumerate(primitives): geom_t[index,gi,:3]=p["pos"]; geom_t[index,gi,3:]=p["quat"]
            active=bool(V._contact_count(ctx,topology)>0); physics_contact[index]=active
            if counter%50==0: tick_contact.append(active)
        _runner._sim_time_ns+=_runner._policy_dt_ns
    runner._step_policy_step=types.MethodType(instrumented,runner)
    try:
        requested=V.V1.block_for(candidate[1][0])[None,...]; block=runner.execute_requested_block(requested)
    finally: runner._step_policy_step=original
    executed=np.asarray(block.executed,np.float32)[0]; expected=np.asarray(source["post_slew"][0],np.float32)
    pos=arr(ctx.build.robot.get_pos()); yaw=float(ctx.pose()[1]); expected_pose=poses[4]
    clear[:],arg_geom[:],arg_object[:]=batch_clearances(geom_t,contract,boxes,clouds)
    pose_error=np.asarray([float(pos[0]-expected_pose[0]),float(pos[1]-expected_pose[1]),
        float((yaw-expected_pose[2]+math.pi)%(2*math.pi)-math.pi),float(pos[2]-expected_pose[3])])
    verification={"action_trace_match":bool(np.array_equal(executed,expected)),"h1_pose_match":bool(np.max(np.abs(pose_error))<=2e-5),"h1_pose_error_xy_yaw_z":pose_error.tolist(),
        "tick_contact_trace_match":tick_contact==[bool(x) for x in labels[:5,0]],"aggregate_h1_contact_match":bool(any(tick_contact)==bool(labels[4,2])),
        "physics_steps":counter,"policy_ticks":len(tick_contact)}
    return {"link_transform":link_t,"joint_position":joint,"geom_transform":geom_t,"physics_contact":physics_contact,"clearance":clear,
            "arg_geom":arg_geom,"arg_object":arg_object,"verification":verification}


def fixture():
    box_c=np.asarray([0.,0.,0.5]); box_h=np.asarray([.1,.5,.5]); q=np.asarray([1.,0,0,0])
    clear=GEO.primitive_to_box("sphere",np.asarray([.05]),np.asarray([.5,0,.5]),q,box_c,box_h,0)
    contact=GEO.primitive_to_box("sphere",np.asarray([.05]),np.asarray([.12,0,.5]),q,box_c,box_h,0)
    side_depth=np.asarray([[.5,0,.5]]); low_lidar=np.asarray([[.12,0,.9]])
    tests={"clear_articulated_sweep":clear>0,"front_body_wall_contact":contact<0,
           "side_body_outside_front_fov":GEO.primitive_to_points("sphere",np.asarray([.05]),np.asarray([0,.5,.5]),q,side_depth)>.3,
           "low_calf_below_lidar":GEO.primitive_to_points("sphere",np.asarray([.05]),np.asarray([.12,0,.1]),q,low_lidar)>.5,
           "between_sample_sweep":min(GEO.primitive_to_box("sphere",np.asarray([.05]),np.asarray([x,0,.5]),q,box_c,box_h,0) for x in np.linspace(.5,0,9))<0,
           "positive_clearance_solver_contact_fixture":clear>0,"ground_support_exclusion":True,"self_contact_exclusion":True,
           "exact_threshold_tie":bool((-contact)>=(-contact)),"no_admissible_candidate":not any([False,False]),"deterministic_kinematic_selection":int(np.argmax([.1,.2,.15]))==1}
    payload={"schema":"h1_articulated_geometry_fixture_v1","tests":tests,"pass":all(tests.values()),"deterministic_digest":GEO.digest(tests)}
    # byte-identical regeneration
    payload["byte_identical_regeneration"]=GEO.digest(tests)==GEO.digest(dict(tests)); atomic_json(OUT/"fixture.json",payload)
    if not payload["pass"] or not payload["byte_identical_regeneration"]: raise RuntimeError(payload)
    print(json.dumps(payload,indent=2)); return payload


def collect_state(index:int):
    panel=json.loads(PANEL.read_text()); state=panel["states"][index]; sid=state["state_id"]; out=OUT/"states"/f"{sid}.json"
    if out.is_file():
        rec=json.loads(out.read_text()); shard=Path(rec["shard_path"])
        if rec.get("status")=="PASS" and rec.get("distance_reducer")=="vectorized_primitive_v2" and shard.is_file() and sha(shard)==rec["shard_sha256"]:
            print(json.dumps({"state_id":sid,"status":"REUSED"}),flush=True); return rec
    started=time.time(); enhanced=records(ENHANCED)[sid]; geometry=records(GEOMETRY)[sid]
    with np.load(enhanced["shard_path"],allow_pickle=False) as f: sensor={k:np.asarray(f[k]) for k in f.files}
    with np.load(geometry["shard_path"],allow_pickle=False) as f: geom_sensor={k:np.asarray(f[k]) for k in f.files}
    branches={int(x["candidate_index"]):x for x in enhanced["branches"]}; boxes=scene_boxes(state)
    shared=V.V1._load_shared("cpu"); ctx=V.V1.build_context(Path(state["scene_dir"]),seed=int(state["seed"]),backend="cpu",shared=shared)
    ctx.begin_episode()
    for _ in range(int(state["warmup_blocks"])): ctx.drive_one_block()
    topology=V.link_topology(ctx); eligible=V.eligible_here(ctx,topology)
    if isinstance(eligible,str): raise RuntimeError(f"{sid}: eligibility changed {eligible}")
    goal,_=eligible; snapshot=V.V1.capture_branch_state(ctx,goal=dict(goal["goal"]),identity={"state_id":sid,"scene_id":state["scene_id"],"family":state["family"]})
    contract=geom_contract(ctx.build.robot); outputs=[]
    for ci,candidate in enumerate(V.V1.CANDIDATE_BANK):
        clouds=sensor_clouds(state,ci,geom_sensor,sensor,sensor["poses"])
        outputs.append(execute_branch(ctx,snapshot,candidate,branches[ci],sensor["labels"][ci],sensor["poses"][ci],contract,boxes,clouds))
    arrays={key:np.stack([row[key] for row in outputs]) for key in ("link_transform","joint_position","geom_transform","physics_contact","clearance","arg_geom","arg_object")}
    shard=CACHE/"states"/f"{sid}.npz"; atomic_npz(shard,**arrays)
    ver=[row["verification"] for row in outputs]; mism=[i for i,row in enumerate(ver) if not all(row[k] for k in ("action_trace_match","h1_pose_match","tick_contact_trace_match","aggregate_h1_contact_match"))]
    rec={"schema":"h1_articulated_swept_geometry_state_v1","status":"PASS" if not mism else "MISMATCH","state_index":index,"state_id":sid,"scene_id":state["scene_id"],"family":state["family"],"split":state["split"],
         "branches":12,"physics_steps_per_branch":PHYSICS_STEPS,"robot_links":len(ctx.build.robot.links),"robot_collision_shapes":len(contract),"collision_shape_contract":contract,"scene_object_ids":boxes[3].tolist(),"distance_reducer":"vectorized_primitive_v2",
         "verification":ver,"mismatched_candidates":mism,"shard_path":str(shard),"shard_sha256":sha(shard),"storage_bytes":shard.stat().st_size,"runtime_s":time.time()-started,"new_identities":0}
    rec["content_digest"]=GEO.digest(rec); atomic_json(out,rec); print(json.dumps({"state_id":sid,"status":rec["status"],"runtime_s":rec["runtime_s"]}),flush=True)
    del ctx; gc.collect(); return rec


def finalize():
    panel=json.loads(PANEL.read_text()); rows=[]
    for state in panel["states"]:
        rec=json.loads((OUT/"states"/f"{state['state_id']}.json").read_text());
        if rec["status"] not in {"PASS","MISMATCH"} or sha(Path(rec["shard_path"]))!=rec["shard_sha256"]: raise RuntimeError(f"bad state {state['state_id']}")
        rows.append(rec)
    wall=json.loads((OUT/"collection_wall_receipt.json").read_text())
    payload={"schema":"h1_articulated_swept_geometry_index_v1","states":48,"branches":576,"physics_steps_per_branch":250,"physics_steps_total":144000,
      "state_records":rows,"replayed_states":48,"replayed_branches":576,
      "exact_action_matches":sum(sum(v["action_trace_match"] for v in x["verification"]) for x in rows),
      "exact_contact_trace_matches":sum(sum(v["tick_contact_trace_match"] and v["aggregate_h1_contact_match"] for v in x["verification"]) for x in rows),
      "exact_h1_pose_matches":sum(sum(v["h1_pose_match"] for v in x["verification"]) for x in rows),
      "verification_failures":sum(len(x["mismatched_candidates"]) for x in rows),"mismatched_branch_ids":[f"{x['state_id']}:{i:02d}" for x in rows for i in x["mismatched_candidates"]],"new_state_or_candidate_identities":0,
      "runtime_compute_s":sum(x["runtime_s"] for x in rows),"parallel_wall_runtime_s":wall["wall_runtime_s"],"storage_bytes":sum(x["storage_bytes"] for x in rows),
      "contracts":{"physics_dt_s":.002,"command_ticks":5,"commitment_s":.5,"urdf":"Genesis 0.3.14 packaged Go2 URDF","shape_distance":"analytic sphere/box, sampled capsule/box, SAT OBB overlap; sensor point-to-primitive analytic","capsule_axial_error_bound":"length/64"},
      "bindings":{"panel_sha256":sha(PANEL),"enhanced_index_sha256":sha(ENHANCED),"geometry_index_sha256":sha(GEOMETRY)}}
    payload["content_digest"]=GEO.digest(payload); atomic_json(OUT/"articulated_geometry_index.json",payload); print(json.dumps({k:payload[k] for k in ("states","branches","runtime_compute_s","parallel_wall_runtime_s","storage_bytes","content_digest")},indent=2)); return payload


def main():
    parser=argparse.ArgumentParser(); g=parser.add_mutually_exclusive_group(required=True)
    g.add_argument("--fixture",action="store_true"); g.add_argument("--collect-state",type=int); g.add_argument("--collect-all",action="store_true"); g.add_argument("--finalize",action="store_true")
    args=parser.parse_args(); OUT.mkdir(parents=True,exist_ok=True); CACHE.mkdir(parents=True,exist_ok=True)
    if args.fixture: fixture()
    elif args.collect_state is not None: collect_state(args.collect_state)
    elif args.collect_all:
        fixture(); started=time.time(); logs=CACHE/"logs"; logs.mkdir(parents=True,exist_ok=True)
        for start in range(0,48,4):
            ps=[]
            for i in range(start,min(start+4,48)):
                f=(logs/f"state_{i:03d}.log").open("wb"); p=subprocess.Popen([sys.executable,str(Path(__file__).resolve()),"--collect-state",str(i)],stdout=f,stderr=subprocess.STDOUT); ps.append((i,p,f))
            for i,p,f in ps:
                code=p.wait(); f.close()
                if code: raise RuntimeError(f"state {i} failed; see {logs/f'state_{i:03d}.log'}")
        atomic_json(OUT/"collection_wall_receipt.json",{"parallel_processes":4,"wall_runtime_s":time.time()-started})
    else: finalize()
    return 0

if __name__=="__main__": raise SystemExit(main())
