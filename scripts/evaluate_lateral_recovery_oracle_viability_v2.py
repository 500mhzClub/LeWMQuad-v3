#!/usr/bin/env python3
"""Oracle 14-action viability evaluation for the qualified lateral controller."""
from __future__ import annotations

import argparse
import copy
import gc
import hashlib
import json
import math
import os
import subprocess
from pathlib import Path
import sys
import time

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
for extra in (ROOT,ROOT/"scripts",ROOT/"lewm_genesis",ROOT/"lewm_worlds"):
    if str(extra) not in sys.path:sys.path.insert(0,str(extra))

from lewm.control import lateral_controller_failure_attribution_v2 as C
from scripts import materialize_genesis_narrowphase_candidate_feasibility_v1 as NARROW
from scripts import materialize_one_tick_viability_constrained_mpc_v1 as ONE
from scripts import materialize_multi_cycle_viability_envelope_v1 as MULTI
from scripts import run_go2_oracle_branch_pilot_v1_2 as V
from lewm_genesis.rollout import GenesisGo2PPOPolicy

OUT=ROOT/".generated/lateral_controller_failure_attribution_and_full_budget_successor_v2"
CACHE=Path.home()/".cache/lewm_go2_temporal_v03/lateral_controller_failure_attribution_and_full_budget_successor_v2"
PANEL=ROOT/".generated/wide_geometry_embodied_contact_proxy_v1/fresh_panel_manifest.json"
OLD_TREE=ROOT/".generated/one_tick_viability_constrained_mpc_v1/viability_tree_index.json"
OLD_RESULT=ROOT/".generated/one_tick_viability_constrained_mpc_v1/viability_result.json"
OLD_MULTI=ROOT/".generated/multi_cycle_viability_envelope_v1/multi_cycle_index.json"
SELECTION=ROOT/".generated/multi_cycle_viability_envelope_v1/frozen_state_selection.json"
CHECKPOINT=OUT/"seed_2026082015/model_999.pt"
CFG=OUT/"seed_2026082015/cfgs.pkl"
PHYSICS_STEPS=50
MAX_CYCLES=10
LATERAL_ACTIONS=((12,"lateral_left",(0.0,0.20,0.0)),(13,"lateral_right",(0.0,-0.20,0.0)))


def sha(path:Path)->str:
    h=hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda:f.read(1<<20),b""):h.update(b)
    return h.hexdigest()


def atomic_json(path:Path,value):
    path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_name(f".{path.name}.tmp-{os.getpid()}")
    tmp.write_text(json.dumps(value,indent=2,sort_keys=True,allow_nan=False)+"\n");os.replace(tmp,path)


def lateral_policy():
    return GenesisGo2PPOPolicy(checkpoint_path=CHECKPOINT,cfg_path=CFG,device="cpu")


def topology(ctx):
    topo=V.link_topology(ctx);solver=ctx.build.scene.rigid_solver
    links={int(link.idx):str(link.name) for link in solver.links};objects={int(link.idx):str(link.entity.name) for link in solver.links}
    return topo,links,objects


def execute_lateral(ctx,snapshot,policy,action_index,name,command,topo,links,objects,identity):
    ctx.policy=policy;ONE._restore_tick_boundary(ctx,snapshot);runner=ctx.runner
    target=np.asarray(command,np.float32)[None,:]
    contact=np.zeros(PHYSICS_STEPS,np.uint8);first=None;counter=0
    start=ctx.pose()
    for _ in range(int(runner._policy_steps_per_command_tick)):
        observation=runner._build_observation(target);joint_targets=runner.policy.act(observation);runner._apply_joint_targets(joint_targets)
        for _ in range(int(runner._physics_steps_per_policy)):
            runner.build.scene.step();summary=NARROW.contact_summary(runner.build.robot.get_contacts(),topo,links,objects,force_threshold=True)
            contact[counter]=bool(summary["active"])
            if summary["active"] and first is None:first=summary
            counter+=1
        runner._sim_time_ns+=runner._policy_dt_ns
    if counter!=PHYSICS_STEPS:raise RuntimeError(counter)
    ONE._advance_tick_counters(ctx,target);end=ctx.pose();flags=V.V1._termination_flags(ctx)
    successor=ONE._capture_tick_boundary(ctx,goal=snapshot.goal,identity={**snapshot.identity,"branch_identity":identity,"active_controller":"lateral"})
    dx=float(end[0][0]-start[0][0]);dy=float(end[0][1]-start[0][1]);yaw=float(start[1])
    return {"action_index":action_index,"candidate":name,"controller":"lateral_successor","target_command":list(command),"first_tick_contact":bool(contact.any()),
            "first_contact_step":None if not contact.any() else int(np.flatnonzero(contact)[0]),"contact_link":None if first is None else first.get("robot_link_name"),
            "contact_object":None if first is None else first.get("environment_object"),"endpoint_pose":[[float(end[0][0]),float(end[0][1])],float(end[1]),float(end[2])],
            "body_displacement_xy_m":[math.cos(yaw)*dx+math.sin(yaw)*dy,-math.sin(yaw)*dx+math.cos(yaw)*dy],"termination":flags,
            "successor_digest":successor.digest},contact,successor


def execute_route(ctx,snapshot,route_policy,index,topo,links,objects):
    ctx.policy=route_policy
    record,contact,_r,_o,successor=ONE._execute_one_tick(ctx,snapshot,index,topo,links,objects)
    record={**record,"action_index":index,"controller":"frozen_route","first_tick_contact":bool(contact.any())}
    return record,contact,successor


def successor_outcomes(ctx,snapshot,route_policy,lat_policy,topo,links,objects,identity):
    rows=[]
    for index in range(12):
        record,contact,_successor=execute_route(ctx,snapshot,route_policy,index,topo,links,objects)
        rows.append({"action_index":index,"candidate":record["candidate"],"controller":"frozen_route","contact":bool(contact.any()),"first_contact_step":record["first_contact_step"]})
    for index,name,command in LATERAL_ACTIONS:
        record,contact,_successor=execute_lateral(ctx,snapshot,lat_policy,index,name,command,topo,links,objects,f"{identity}:next:{index}")
        rows.append({"action_index":index,"candidate":name,"controller":"lateral_successor","contact":bool(contact.any()),"first_contact_step":record["first_contact_step"]})
    return rows


def route_plan(ctx,snapshot,route_policy,index,route):
    ctx.policy=route_policy;return MULTI._h3_plan(ctx,snapshot,index,route)


def augmented_tree(ctx,snapshot,route_policy,lat_policy,identity):
    topo,links,objects=topology(ctx);ctx.policy=route_policy;route=MULTI._route_contract(ctx,snapshot)
    start=route["pose"];start_dist=math.hypot(route["waypoint_xy"][0]-start[0][0],route["waypoint_xy"][1]-start[0][1])
    start_heading=abs(MULTI.KINEMATIC.wrap(route["route_heading_world_rad"]-start[1]));rows=[];successors={};branches=0
    for index in range(14):
        if index<12:
            current,contact,successor=execute_route(ctx,snapshot,route_policy,index,topo,links,objects);current.update(route_plan(ctx,snapshot,route_policy,index,route))
        else:
            _idx,name,command=LATERAL_ACTIONS[index-12];current,contact,successor=execute_lateral(ctx,snapshot,lat_policy,index,name,command,topo,links,objects,f"{identity}:current:{index}")
            current.update({"h3_progress_m":None,"h3_heading_improvement_rad":None})
        safe=not bool(contact.any());next_rows=[]
        if safe:
            successors[index]=successor;next_rows=successor_outcomes(ctx,successor,route_policy,lat_policy,topo,links,objects,f"{identity}:succ:{index}");branches+=14
        safe_next=[r["action_index"] for r in next_rows if not r["contact"]]
        endpoint=current["endpoint_pose"];end_dist=math.hypot(route["waypoint_xy"][0]-endpoint[0][0],route["waypoint_xy"][1]-endpoint[0][1]);end_heading=abs(MULTI.KINEMATIC.wrap(route["route_heading_world_rad"]-endpoint[1]))
        rows.append({**current,"safe_prefix":safe,"successor_rows":next_rows,"successor_safe_action_indices":safe_next,"successor_safe_action_count":len(safe_next),"viable":bool(safe_next),"admissible":bool(safe and safe_next),
                     "immediate_progress_m":float(start_dist-end_dist),"immediate_heading_improvement_rad":float(start_heading-end_heading)})
    return {"identity":identity,"route":route,"candidates":rows,"safe_prefix_count":sum(r["safe_prefix"] for r in rows),"viability_admissible_count":sum(r["admissible"] for r in rows),"current_prefix_branches":14,"successor_branches":branches},successors


def full_panel_state(index:int):
    panel=json.loads(PANEL.read_text())["states"];state=panel[index];state_id=state["state_id"];path=OUT/"scientific_states"/f"{state_id}.json"
    if path.is_file():return json.loads(path.read_text())
    old_records={r["state_id"]:r for r in json.loads(OLD_TREE.read_text())["state_records"]};old=old_records[state_id]
    started=time.time();ctx,snapshot=ONE._build_current(state);route_policy=ctx.policy;lat=lateral_policy();topo,links,objects=topology(ctx);augmented=[];branches=0
    for old_row in old["current"]:
        row=copy.deepcopy(old_row);row["action_index"]=int(row["candidate_index"]);row["first_tick_contact"]=bool(row["contact"]);safe=bool(row["safe_prefix"])
        lateral_next=[]
        if safe:
            current,contact,successor=execute_route(ctx,snapshot,route_policy,int(row["candidate_index"]),topo,links,objects);branches+=1
            if bool(contact.any())!=bool(row["contact"]):raise RuntimeError(f"{state_id}: historical replay mismatch")
            for action_index,name,command in LATERAL_ACTIONS:
                outcome,c,_=execute_lateral(ctx,successor,lat,action_index,name,command,topo,links,objects,f"{state_id}:route:{row['candidate_index']}:next:{action_index}");branches+=1
                lateral_next.append({"action_index":action_index,"candidate":name,"contact":bool(c.any()),"first_contact_step":outcome["first_contact_step"]})
        added=sum(not r["contact"] for r in lateral_next);row["historical_successor_safe_action_count"]=int(row["successor_safe_candidate_count"]);row["lateral_successor_rows"]=lateral_next
        row["successor_safe_action_count"]=row["historical_successor_safe_action_count"]+added;row["viable"]=row["successor_safe_action_count"]>0;row["admissible"]=safe and row["viable"];augmented.append(row)
    for action_index,name,command in LATERAL_ACTIONS:
        current,contact,successor=execute_lateral(ctx,snapshot,lat,action_index,name,command,topo,links,objects,f"{state_id}:current:{action_index}");branches+=1;safe=not bool(contact.any());next_rows=[]
        if safe:next_rows=successor_outcomes(ctx,successor,route_policy,lat,topo,links,objects,f"{state_id}:lateral:{action_index}");branches+=14
        safe_next=[r["action_index"] for r in next_rows if not r["contact"]]
        augmented.append({**current,"safe_prefix":safe,"successor_rows":next_rows,"successor_safe_action_count":len(safe_next),"successor_safe_action_indices":safe_next,"viable":bool(safe_next),"admissible":bool(safe and safe_next)})
    result={"schema":"lateral_recovery_full_panel_state_v2","state_id":state_id,"family":state["family"],"split":state["split"],"historical_viability":any(bool(r["safe_prefix"] and r["viable"]) for r in old["current"]),
            "augmented_viability":any(bool(r["admissible"]) for r in augmented),"historical_admissible_count":sum(bool(r["safe_prefix"] and r["viable"]) for r in old["current"]),"augmented_admissible_count":sum(bool(r["admissible"]) for r in augmented),
            "lateral_admissible_indices":[r["action_index"] for r in augmented if r["action_index"]>=12 and r["admissible"]],"actions":augmented,"generated_branches":branches,"runtime_s":time.time()-started}
    result["content_digest"]=C.digest(result);atomic_json(path,result);del ctx;gc.collect();print(json.dumps({"state":state_id,"before":result["historical_viability"],"after":result["augmented_viability"],"lateral":result["lateral_admissible_indices"],"branches":branches}),flush=True);return result


def choose(tree):
    route=[r for r in tree["candidates"] if r["action_index"]<12 and r["admissible"]]
    if route:
        order=MULTI.REDUCE.route_order(route);return route[order[0]]
    lateral=[r for r in tree["candidates"] if r["action_index"]>=12 and r["admissible"]]
    if not lateral:return None
    # Genesis 0.3.14 exposes exact penetration/contact but not positive exact
    # clearance. Safe-count decides first; action index is the frozen final tie.
    return min(lateral,key=lambda r:(-r["successor_safe_action_count"],r["action_index"]))


def run_rollout(state_id):
    selection=json.loads(SELECTION.read_text());multi={r["state_id"]:r for r in json.loads(OLD_MULTI.read_text())["state_records"]};old=multi[state_id]
    panel={r["state_id"]:r for r in json.loads(PANEL.read_text())["states"]};state=panel[state_id];role=old["role"];started=time.time()
    if role=="failure" and old["stable_predecessor_depth"] is not None:
        ctx,snapshots,_=MULTI.historical_snapshots(state);depth=int(old["stable_predecessor_depth"]);snapshot=snapshots[depth]
    else:
        ctx,snapshot=ONE._build_current(state);depth=0
    route_policy=ctx.policy;lat=lateral_policy();selected=[];generated=0
    for cycle in range(MAX_CYCLES):
        tree,snaps=augmented_tree(ctx,snapshot,route_policy,lat,f"{state_id}:roll:{cycle:02d}");generated+=tree["current_prefix_branches"]+tree["successor_branches"];choice=choose(tree)
        if choice is None:selected.append({"cycle":cycle,"abstained":True,"tree":tree});break
        snapshot=snaps[choice["action_index"]];selected.append({"cycle":cycle,"abstained":False,"selected_action_index":choice["action_index"],"selected_action":choice["candidate"],"selected_controller":choice["controller"],
            "selected_first_tick_contact":choice["first_tick_contact"],"selected_successor_viable":choice["viable"],"selected_successor_safe_action_count":choice["successor_safe_action_count"],
            "immediate_progress_m":choice["immediate_progress_m"],"h3_progress_m":choice.get("h3_progress_m"),"termination":choice["termination"],"tree":tree})
        if choice["first_tick_contact"] or any(choice["termination"].values()):break
    result={"schema":"lateral_recovery_multi_cycle_rollout_v2","state_id":state_id,"family":state["family"],"role":role,"start_depth":depth,"selected":selected,"completed_cycles":sum(not r["abstained"] for r in selected),
            "selected_contacts":sum(bool(r.get("selected_first_tick_contact")) for r in selected),"selected_nonviable":sum(r.get("selected_successor_viable") is False for r in selected),
            "lateral_left_selections":sum(r.get("selected_action_index")==12 for r in selected),"lateral_right_selections":sum(r.get("selected_action_index")==13 for r in selected),"route_selections":sum(r.get("selected_action_index") is not None and int(r["selected_action_index"])<12 for r in selected if not r["abstained"]),
            "controller_switches":sum(selected[i].get("selected_controller")!=selected[i-1].get("selected_controller") for i in range(1,len(selected)) if not selected[i]["abstained"] and not selected[i-1]["abstained"]),
            "cycles_with_two_safe_successors":sum(int(r.get("selected_successor_safe_action_count",0)>=2) for r in selected),"distance_progress_m":sum(float(r.get("immediate_progress_m",0)) for r in selected),
            "temporary_negative_progress_cycles":sum(float(r.get("immediate_progress_m",0))<0 for r in selected if not r["abstained"]),"falls_or_unsafe":sum(any(r.get("termination",{}).values()) for r in selected),"generated_branches":generated,"runtime_s":time.time()-started}
    result["content_digest"]=C.digest(result);atomic_json(OUT/"scientific_rollouts"/f"{state_id}.json",result);del ctx;gc.collect();print(json.dumps({"rollout":state_id,"cycles":result["completed_cycles"],"lateral":result["lateral_left_selections"]+result["lateral_right_selections"],"contact":result["selected_contacts"],"branches":generated}),flush=True);return result


def collect_all():
    started=time.time();logs=CACHE/"logs";logs.mkdir(parents=True,exist_ok=True)
    for start in range(0,48,4):
        jobs=[]
        for index in range(start,min(start+4,48)):
            log=logs/f"panel_{index:02d}.log";stream=log.open("wb");p=subprocess.Popen([sys.executable,str(Path(__file__).resolve()),"--state",str(index)],stdout=stream,stderr=subprocess.STDOUT);jobs.append((index,p,stream,log))
        for index,p,stream,log in jobs:
            code=p.wait();stream.close()
            if code:raise RuntimeError(f"panel state {index} failed; see {log}")
    selection=json.loads(SELECTION.read_text());ids=selection["failure_state_ids"]+selection["control_state_ids"]
    for start in range(0,len(ids),4):
        jobs=[]
        for state_id in ids[start:start+4]:
            log=logs/f"roll_{state_id}.log";stream=log.open("wb");p=subprocess.Popen([sys.executable,str(Path(__file__).resolve()),"--rollout",state_id],stdout=stream,stderr=subprocess.STDOUT);jobs.append((state_id,p,stream,log))
        for state_id,p,stream,log in jobs:
            code=p.wait();stream.close()
            if code:raise RuntimeError(f"rollout {state_id} failed; see {log}")
    atomic_json(OUT/"scientific_collection_receipt.json",{"wall_runtime_s":time.time()-started,"states":48,"rollouts":16,"parallel_processes":4})


def finalize():
    panel=json.loads(PANEL.read_text())["states"];states=[json.loads((OUT/"scientific_states"/f"{s['state_id']}.json").read_text()) for s in panel]
    selection=json.loads(SELECTION.read_text());ids=selection["failure_state_ids"]+selection["control_state_ids"]
    roll=[json.loads((OUT/"scientific_rollouts"/f"{s}.json").read_text()) for s in ids];old_multi={r["state_id"]:r for r in json.loads(OLD_MULTI.read_text())["state_records"]}
    cycles=[x for r in roll for x in r["selected"] if not x["abstained"]];controls=[r for r in roll if r["role"]=="matched_control"]
    old_control_progress=sum(float(old_multi[r["state_id"]]["multi_cycle_rollout"]["distance_progress_m"]) for r in controls);new_control_progress=sum(float(r["distance_progress_m"]) for r in controls)
    full_after=sum(r["augmented_viability"] for r in states);persistent=next(r for r in roll if r["state_id"]=="wide-held-2-04")
    stable_ids={state_id for state_id in selection["failure_state_ids"] if old_multi[state_id]["stable_predecessor_depth"] is not None}
    intermittent_ids={"wide-cal-0-05","wide-held-2-00"}
    by_id={r["state_id"]:r for r in roll}
    def stable_cycles(record,minimum):
        return record["completed_cycles"]>=minimum and not any(x["abstained"] for x in record["selected"][:minimum]) and record["selected_contacts"]==0 and record["selected_nonviable"]==0
    failure_outcomes={state_id:{
        "historical_classification":old_multi[state_id]["failure_classification"],
        "start_depth":by_id[state_id]["start_depth"],
        "completed_cycles":by_id[state_id]["completed_cycles"],
        "stable_three_cycle_envelope":stable_cycles(by_id[state_id],3),
        "completed_ten_cycle_rollout":stable_cycles(by_id[state_id],10),
        "lateral_selections":by_id[state_id]["lateral_left_selections"]+by_id[state_id]["lateral_right_selections"],
    } for state_id in selection["failure_state_ids"]}
    gate={"wide_held_2_04_resolved":stable_cycles(persistent,3),
          "both_intermittent_failures_stable":all(stable_cycles(by_id[state_id],3) for state_id in intermittent_ids),
          "five_previously_stable_envelopes_preserved":len(stable_ids)==5 and all(stable_cycles(by_id[state_id],3) for state_id in stable_ids),
          "all_rollouts_zero_contacts":sum(r["selected_contacts"] for r in roll)==0,"zero_nonviable_successors":sum(r["selected_nonviable"] for r in roll)==0,
          "transition_failures_zero":sum(r["falls_or_unsafe"] for r in roll)==0,"cycles_two_safe_successors":sum(r["cycles_with_two_safe_successors"] for r in roll)/max(1,len(cycles))>=.95,
          "matched_control_progress_90pct":new_control_progress/max(old_control_progress,1e-12)>=.90,"lateral_used":sum(r["lateral_left_selections"]+r["lateral_right_selections"] for r in roll)>0,
          "full_panel_95pct":full_after/48>=.95}
    passed=all(gate.values());primary="LATERAL_CONTROLLER_AND_VIABILITY_SIGNAL" if passed else "LATERAL_CONTROLLER_SIGNAL_VIABILITY_NO_GO"
    result={"schema":"lateral_controller_scientific_viability_v2_result","claim_boundary":"oracle simulation-only physics-rate disallowed-contact proxy; no learned safety or physical Go2 qualification",
            "full_panel":{"before":40,"after":full_after,"states":48,"per_family":{f:{"states":sum(r['family']==f for r in states),"viable_after":sum(r['family']==f and r['augmented_viability'] for r in states)} for f in sorted({r['family'] for r in states})}},
            "rollouts":{"states":16,"executed_cycles":len(cycles),"selected_contacts":sum(r["selected_contacts"] for r in roll),"selected_nonviable":sum(r["selected_nonviable"] for r in roll),"lateral_left":sum(r["lateral_left_selections"] for r in roll),"lateral_right":sum(r["lateral_right_selections"] for r in roll),"route_actions":sum(r["route_selections"] for r in roll),
                        "controller_switches":sum(r["controller_switches"] for r in roll),"cycles_two_safe_successors":sum(r["cycles_with_two_safe_successors"] for r in roll),"fraction_two_safe_successors":sum(r["cycles_with_two_safe_successors"] for r in roll)/max(1,len(cycles)),
                        "distance_progress_m":sum(r["distance_progress_m"] for r in roll),"temporary_negative_progress_cycles":sum(r["temporary_negative_progress_cycles"] for r in roll),"matched_control_progress_fraction":new_control_progress/max(old_control_progress,1e-12),"failure_outcomes":failure_outcomes,"per_state":roll},
            "gate":gate,"pass":passed,"primary_classification":primary,"generated_branches":sum(r["generated_branches"] for r in states)+sum(r["generated_branches"] for r in roll),"runtime":json.loads((OUT/"scientific_collection_receipt.json").read_text())}
    result["content_digest"]=C.digest(result);atomic_json(OUT/"scientific_viability_result.json",result);print(json.dumps({"primary":primary,"full_after":full_after,"rollouts":result["rollouts"],"gate":gate,"generated_branches":result["generated_branches"]},indent=2))


def main():
    p=argparse.ArgumentParser();g=p.add_mutually_exclusive_group(required=True);g.add_argument("--state",type=int);g.add_argument("--rollout");g.add_argument("--collect-all",action="store_true");g.add_argument("--finalize",action="store_true");a=p.parse_args()
    if a.state is not None:full_panel_state(a.state)
    elif a.rollout:run_rollout(a.rollout)
    elif a.collect_all:collect_all()
    else:finalize()


if __name__=="__main__":main()
