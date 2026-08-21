#!/usr/bin/env python3
"""Calibrate and evaluate frozen H1 articulated-geometry evidence."""
from __future__ import annotations

from collections import Counter
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import time

import numpy as np

from lewm.safety import articulated_swept_geometry_v1 as GEO
from scripts import evaluate_wide_geometry_score_composition_v1 as BASE

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/".generated/h1_articulated_swept_geometry_sufficiency_v1"
CACHE=Path("/home/andrewknowles/.cache/lewm_go2_temporal_v03/h1_articulated_swept_geometry_sufficiency_v1")
INDEX=OUT/"articulated_geometry_index.json"
PANEL=ROOT/".generated/wide_geometry_embodied_contact_proxy_v1/fresh_panel_manifest.json"
LEDGER=Path("/home/andrewknowles/.cache/lewm_go2_temporal_v03/wide_geometry_embodied_contact_proxy_v1/stage1/row_level_evidence_v1.npz")
CONDITIONS=("FULL_ARTICULATED_SCENE_GEOMETRY","DEPTH_ARTICULATED_SWEEP","LIDAR_ARTICULATED_SWEEP","DEPTH_LIDAR_ARTICULATED_SWEEP")
FAMILIES=BASE.FAMILIES


def sha(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda:f.read(1<<22),b""): h.update(b)
    return h.hexdigest()


def atomic_json(path,value):
    path.parent.mkdir(parents=True,exist_ok=True); tmp=path.with_name(f".{path.name}.tmp-{os.getpid()}")
    tmp.write_text(json.dumps(value,indent=2,sort_keys=True,allow_nan=False)+"\n"); os.replace(tmp,path)


def atomic_npz(path,**arrays):
    path.parent.mkdir(parents=True,exist_ok=True); tmp=path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with tmp.open("wb") as f: np.savez_compressed(f,**arrays)
    os.replace(tmp,path)


def load():
    index=json.loads(INDEX.read_text()); panel=json.loads(PANEL.read_text())
    with np.load(LEDGER,allow_pickle=False) as f: ledger={k:np.asarray(f[k]) for k in f.files}
    mask=np.isin(ledger["split"],["calibration","heldout"])
    selected=np.flatnonzero(mask); ids=[str(ledger["branch_id"][i]) for i in selected]
    evidence={}
    metadata={}
    for record in index["state_records"]:
        with np.load(record["shard_path"],allow_pickle=False) as f:
            for ci in range(12):
                bid=f"{record['state_id']}:{ci:02d}"
                evidence[bid]={"clearance_step":np.asarray(f["clearance"][ci],np.float64),"physics_contact":np.asarray(f["physics_contact"][ci],bool),
                    "arg_geom":np.asarray(f["arg_geom"][ci]),"arg_object":np.asarray(f["arg_object"][ci])}
        metadata[record["state_id"]]=record
    if set(ids)!=set(evidence) or len(ids)!=576: raise RuntimeError("frozen row/evidence alignment mismatch")
    scores=np.asarray([[-float(evidence[bid]["clearance_step"][:,c].min()) for c in range(4)] for bid in ids],np.float64)
    rows=[]
    for local,i in enumerate(selected):
        rows.append({"branch_id":ids[local],"state_id":str(ledger["state_id"][i]),"candidate_index":int(ledger["candidate_index"][i]),"family":str(ledger["family"][i]),
          "split":str(ledger["split"][i]),"hard_contact":bool(ledger["contact_labels"][i,4,1]),"h2_contact":bool(ledger["contact_labels"][i,9,1]),"h3_contact":bool(ledger["contact_labels"][i,14,1]),
          "stuck":bool(ledger["stuck_labels"][i,14,1]),"p_d":float(ledger["p_d"][i]),"p_theta":float(ledger["p_theta"][i]),"kinematic":np.asarray(ledger["kinematic"][i],np.float64),"score":scores[local]})
    return index,panel,ledger,rows,evidence,metadata,scores


def subset(rows,split,condition):
    ci=CONDITIONS.index(condition); chosen=[row for row in rows if row["split"]==split]
    output=[{**row,"score_scalar":float(row["score"][ci])} for row in chosen]
    return output


def evaluate(rows,threshold,include_states=True):
    states=BASE.prepare_states(rows); admitted=np.asarray([row["score_scalar"] < -threshold for row in rows],bool)
    # score=-clearance and risk when clearance<=threshold, hence admit score<-threshold
    metric=BASE.decision_metrics(rows,states,admitted,include_states); families=BASE.family_metrics(rows,states,admitted)
    labels=np.asarray([row["hard_contact"] for row in rows],bool); scores=np.asarray([row["score_scalar"] for row in rows])
    metric.update({"auc":BASE.auc(labels,scores),"average_precision":BASE.average_precision(labels,scores),"threshold_clearance_m":float(threshold),"per_family":families})
    for state_row in metric.get("per_state") or []:
        selected=state_row["selected_candidate"]
        match=next((r for r in rows if r["state_id"]==state_row["state_id"] and r["candidate_index"]==selected),None)
        state_row["selected_h2_continuation_contact"]=None if match is None else bool(match["h2_contact"] and not match["hard_contact"])
        state_row["selected_h3_continuation_contact"]=None if match is None else bool(match["h3_contact"] and not match["h2_contact"])
    metric["selected_h2_continuation_contacts"]=sum(x["selected_h2_continuation_contact"] is True for x in metric.get("per_state") or [])
    metric["selected_h3_continuation_contacts"]=sum(x["selected_h3_continuation_contact"] is True for x in metric.get("per_state") or [])
    return metric,admitted


def select_threshold(rows):
    clearances=-np.asarray([r["score_scalar"] for r in rows]); thresholds=np.concatenate(([np.nextafter(clearances.min(),-np.inf)],np.unique(clearances),[np.nextafter(clearances.max(),np.inf)]))
    eligible=[]
    for threshold in thresholds:
        metric,_=evaluate(rows,float(threshold),False)
        if metric["contact_recall"]>=.95:
            key=(metric["states_retaining_contact_negative"],metric["contact_negative_retention"],-metric["selected_contact_count"],metric["mean_selected_route_progress_m"],
                 -(metric["normalized_route_progress_regret"] if metric["normalized_route_progress_regret"] is not None else 1e9),-metric["false_abstentions"],metric["best_contact_negative_top3"] or -1,-float(threshold))
            eligible.append((key,float(threshold),metric))
    if not eligible: raise RuntimeError("no calibration threshold achieves recall >= .95")
    chosen=max(eligible,key=lambda x:x[0])
    return chosen[1],{"thresholds":len(thresholds),"eligible":len(eligible),"selected_metrics":chosen[2]}


def gate(condition,metric):
    family_ok=all(v["states_retaining_contact_negative"]>0 and v["selected_contact_count"]==0 for v in metric["per_family"].values())
    if condition==CONDITIONS[0]:
        checks={"auc":metric["auc"]>=.99,"ap":metric["average_precision"]>=.99,"recall":metric["contact_recall"]>=.99,"fnr":metric["contact_false_negative_rate"]<=.01,
          "retention":metric["contact_negative_retention"]>=.90,"states":metric["states_retaining_contact_negative"]>=22,"no_only_positive":metric["states_only_contact_positive_admitted"]==0,
          "zero_selected":metric["selected_contact_count"]==0,"false_abstentions":metric["false_abstentions"]<=1,"progress":metric["oracle_progress_fraction"] is not None and metric["oracle_progress_fraction"]>=.95,
          "regret":metric["normalized_route_progress_regret"] is not None and metric["normalized_route_progress_regret"]<=.10,"top3":metric["best_contact_negative_top3"] is not None and metric["best_contact_negative_top3"]>=.90,"families":family_ok}
    else:
        checks={"auc":metric["auc"]>=.90,"ap":metric["average_precision"]>=.90,"recall":metric["contact_recall"]>=.95,"fnr":metric["contact_false_negative_rate"]<=.05,
          "retention":metric["contact_negative_retention"]>=.50,"states":metric["states_retaining_contact_negative"]>=18,"no_only_positive":metric["states_only_contact_positive_admitted"]==0,
          "zero_selected":metric["selected_contact_count"]==0,"false_abstentions":metric["false_abstentions"]<=3,"progress":metric["oracle_progress_fraction"] is not None and metric["oracle_progress_fraction"]>=.80,
          "regret":metric["normalized_route_progress_regret"] is not None and metric["normalized_route_progress_regret"]<=.20,"top3":metric["best_contact_negative_top3"] is not None and metric["best_contact_negative_top3"]>=.75,"families":family_ok}
    return {"passed":all(checks.values()),"checks":checks}


def frontier(rows,condition):
    clearances=-np.asarray([r["score_scalar"] for r in rows]); thresholds=np.concatenate(([np.nextafter(clearances.min(),-np.inf)],np.unique(clearances),[np.nextafter(clearances.max(),np.inf)]))
    fields={k:[] for k in ("threshold","recall","fnr","retention","states","selected_contacts","progress","regret","top3","passes_gate")}; passes=[]
    for threshold in thresholds:
        metric,_=evaluate(rows,float(threshold),False); metric["per_family"]=BASE.family_metrics(rows,BASE.prepare_states(rows),np.asarray([r["score_scalar"] < -threshold for r in rows]))
        g=gate(condition,metric)
        vals=(threshold,metric["contact_recall"],metric["contact_false_negative_rate"],metric["contact_negative_retention"],metric["states_retaining_contact_negative"],metric["selected_contact_count"],metric["mean_selected_route_progress_m"],metric["normalized_route_progress_regret"],metric["best_contact_negative_top3"],g["passed"])
        for key,value in zip(fields,vals): fields[key].append(np.nan if value is None else value)
        if g["passed"]: passes.append((threshold,metric))
    arrays={k:np.asarray(v) for k,v in fields.items()}; path=CACHE/"frontiers"/f"{condition.lower()}_heldout.npz"; atomic_npz(path,**arrays)
    safety=np.asarray(arrays["recall"])>=.95; zero=np.asarray(arrays["selected_contacts"])==0
    def finite_best(values,mask,mode="max"):
        values=np.asarray(values,float)[mask&np.isfinite(values)]; return None if not len(values) else float(values.max() if mode=="max" else values.min())
    return {"thresholds":len(thresholds),"complete_gate_points":len(passes),"any_complete_gate":bool(passes),"maximum_retention_at_recall_ge_0_95":finite_best(arrays["retention"],safety),
      "maximum_retained_states_at_recall_ge_0_95":finite_best(arrays["states"],safety),"maximum_progress_with_zero_selected_contact":finite_best(arrays["progress"],zero),
      "minimum_regret_under_recall_and_zero_selection":finite_best(arrays["regret"],safety&zero,"min"),"path":str(path),"sha256":sha(path)}


def attribution(rows,evidence,metadata,thresholds):
    output=[]; causes=Counter(); body=Counter(); visibility=Counter(); by_family=Counter()
    for row in rows:
        if not row["hard_contact"]: continue
        ev=evidence[row["branch_id"]]; full_step=int(np.argmin(ev["clearance_step"][:,0])); pc=np.flatnonzero(ev["physics_contact"]); first=int(pc[0]) if len(pc) else None
        state_meta=metadata[row["state_id"]]; contract=state_meta["collision_shape_contract"]
        gi=int(ev["arg_geom"][full_step,0]); oi=int(ev["arg_object"][full_step]); link=contract[gi]["link_name"] if gi>=0 else "unresolved"
        if link=="base": region="trunk"
        elif link.startswith(("FL","FR")): region="front_limb"
        elif link.startswith(("RL","RR")): region="rear_limb"
        else: region="unresolved"
        mins=ev["clearance_step"].min(0); seen=[bool(mins[i]<=thresholds[CONDITIONS[i]]) for i in range(1,4)]
        support="both" if seen[0] and seen[1] else "depth" if seen[0] else "lidar" if seen[1] else "neither"
        if not seen[0] and seen[1]: cause="front_depth_fov_or_occlusion"
        elif seen[0] and not seen[1]: cause="lidar_vertical_coverage"
        elif not seen[0] and not seen[1]: cause="sensor_occlusion_timing_or_reconstruction"
        else: cause="geometrically_observed"
        causes[cause]+=1; body[region]+=1; visibility[support]+=1; by_family[(row["family"],cause)]+=1
        output.append({"branch_id":row["branch_id"],"family":row["family"],"candidate_index":row["candidate_index"],"first_contact_physics_step":first,
          "minimum_clearance_physics_step":full_step,"robot_link":link,"body_region":region,"contact_object":None if oi<0 else state_meta["scene_object_ids"][oi],
          "minimum_clearance_m":{CONDITIONS[i]:float(mins[i]) for i in range(4)},"sensor_support":support,"attribution":cause,
          "later_h2_contact":row["h2_contact"] and not row["hard_contact"],"later_h3_contact":row["h3_contact"] and not row["h2_contact"]})
    return {"contacts":len(output),"body_region":dict(body),"sensor_support":dict(visibility),"causes":dict(causes),
      "by_family_cause":{f"{f}/{c}":n for (f,c),n in sorted(by_family.items())},"rows":output}


def physics_comparison(rows,evidence):
    branch_labels=np.asarray([r["hard_contact"] for r in rows],bool); any_physics=np.asarray([evidence[r["branch_id"]]["physics_contact"].any() for r in rows])
    return {"branches":len(rows),"frozen_h1_positive":int(branch_labels.sum()),"physics_step_any_positive":int(any_physics.sum()),"exact_branch_agreement":int((branch_labels==any_physics).sum()),
      "frozen_positive_without_physics_contact":int((branch_labels&~any_physics).sum()),"physics_transient_without_frozen_tick_positive":int((~branch_labels&any_physics).sum()),
      "physics_steps":len(rows)*250,"positive_physics_steps":int(sum(evidence[r["branch_id"]]["physics_contact"].sum() for r in rows))}


def main():
    started=time.time(); index,panel,ledger,rows,evidence,metadata,scores=load(); fixture=json.loads((OUT/"fixture.json").read_text())
    calibration={}; heldout={}; frontiers={}; thresholds={}
    for condition in CONDITIONS:
        cal=subset(rows,"calibration",condition); threshold,selection=select_threshold(cal); thresholds[condition]=threshold
        cm,_=evaluate(cal,threshold,True); hm,_=evaluate(subset(rows,"heldout",condition),threshold,True); hm["gate"]=gate(condition,hm)
        calibration[condition]={"threshold_clearance_m":threshold,"frontier":selection,"metrics":cm}; heldout[condition]=hm
        frontiers[condition]=frontier(subset(rows,"heldout",condition),condition)
    full_pass=heldout[CONDITIONS[0]]["gate"]["passed"]
    if not full_pass: classification="FULL_ARTICULATED_GEOMETRY_CONTACT_PROXY_NO_GO"
    elif heldout[CONDITIONS[1]]["gate"]["passed"]: classification="DEPTH_SWEPT_GEOMETRY_SIGNAL"
    elif heldout[CONDITIONS[2]]["gate"]["passed"]: classification="LIDAR_SWEPT_GEOMETRY_SIGNAL"
    elif heldout[CONDITIONS[3]]["gate"]["passed"]: classification="FUSED_SWEPT_GEOMETRY_SIGNAL"
    else: classification="SENSOR_GEOMETRY_COVERAGE_NO_GO"
    secondary=[]
    if heldout[CONDITIONS[1]]["auc"]<heldout[CONDITIONS[2]]["auc"]: secondary.append("DEPTH_FIELD_OF_VIEW_LIMITATION")
    if heldout[CONDITIONS[2]]["auc"]<heldout[CONDITIONS[3]]["auc"]: secondary.append("LIDAR_VERTICAL_COVERAGE_LIMITATION")
    if full_pass: secondary.append("ARTICULATED_LINK_GEOMETRY_REQUIRED")
    if physics_comparison(subset(rows,"heldout",CONDITIONS[0]),evidence)["physics_transient_without_frozen_tick_positive"]>0: secondary.append("SENSOR_TIMING_LIMITATION")
    if not full_pass: secondary.append("DYNAMIC_CONTACT_NOT_PURELY_GEOMETRIC")
    if any(m["selected_h2_continuation_contacts"]+m["selected_h3_continuation_contacts"] for m in heldout.values()): secondary.append("CONTINUATION_CONTACT_RISK")
    held_rows=subset(rows,"heldout",CONDITIONS[0]); attr=attribution(held_rows,evidence,metadata,thresholds)
    mismatch_ids=set(index.get("mismatched_branch_ids",[])); sensitivity={}
    for condition in CONDITIONS:
        cal=[r for r in subset(rows,"calibration",condition) if r["branch_id"] not in mismatch_ids]
        test=[r for r in subset(rows,"heldout",condition) if r["branch_id"] not in mismatch_ids]
        threshold,_selection=select_threshold(cal); metric,_=evaluate(test,threshold,True); metric["gate"]=gate(condition,metric)
        sensitivity[condition]={"excluded_mismatched_branches":sorted(mismatch_ids),"threshold_clearance_m":threshold,
          "heldout_auc":metric["auc"],"heldout_average_precision":metric["average_precision"],"heldout_recall":metric["contact_recall"],
          "heldout_retention":metric["contact_negative_retention"],"gate_passed":metric["gate"]["passed"]}
    evidence_path=CACHE/"row_level_evidence_v1.npz"; row_ids=np.asarray([r["branch_id"] for r in rows]); splits=np.asarray([r["split"] for r in rows]); labels=np.asarray([r["hard_contact"] for r in rows],np.uint8)
    admissions=np.stack([scores[:,i] < -thresholds[c] for i,c in enumerate(CONDITIONS)],1).astype(np.uint8)
    selected=np.zeros((len(rows),4),np.uint8); selected_pairs=set()
    for ci,condition in enumerate(CONDITIONS):
        for split in ("calibration","heldout"):
            metric=calibration[condition]["metrics"] if split=="calibration" else heldout[condition]
            for state_row in metric["per_state"]:
                if state_row["selected_candidate"] is not None: selected_pairs.add((ci,split,state_row["state_id"],int(state_row["selected_candidate"])))
    link_name=[]; body_region=[]; visibility=[]
    for row in rows:
        ev=evidence[row["branch_id"]]; step=int(np.argmin(ev["clearance_step"][:,0])); gi=int(ev["arg_geom"][step,0]); contract=metadata[row["state_id"]]["collision_shape_contract"]
        link=contract[gi]["link_name"] if gi>=0 else "unresolved"; link_name.append(link)
        body_region.append("trunk" if link=="base" else "front_limb" if link.startswith(("FL","FR")) else "rear_limb" if link.startswith(("RL","RR")) else "unresolved")
        seen_depth=bool(ev["clearance_step"][:,1].min()<=thresholds[CONDITIONS[1]])
        seen_lidar=bool(ev["clearance_step"][:,2].min()<=thresholds[CONDITIONS[2]])
        visibility.append("both" if seen_depth and seen_lidar else "depth" if seen_depth else "lidar" if seen_lidar else "neither")
    for i,row in enumerate(rows):
        for ci in range(4): selected[i,ci]=(ci,row["split"],row["state_id"],row["candidate_index"]) in selected_pairs
    atomic_npz(evidence_path,branch_id=row_ids,state_id=np.asarray([r["state_id"] for r in rows]),candidate_index=np.asarray([r["candidate_index"] for r in rows],np.int16),
      family=np.asarray([r["family"] for r in rows]),split=splits,score=scores.astype(np.float32),h1_contact=labels,h2_contact=np.asarray([r["h2_contact"] for r in rows],np.uint8),
      h3_contact=np.asarray([r["h3_contact"] for r in rows],np.uint8),stuck=np.asarray([r["stuck"] for r in rows],np.uint8),thresholds=np.asarray([thresholds[c] for c in CONDITIONS]),
      admitted=admissions,selected=selected,p_d=np.asarray([r["p_d"] for r in rows],np.float32),p_theta=np.asarray([r["p_theta"] for r in rows],np.float32),
      kinematic=np.stack([r["kinematic"] for r in rows]).astype(np.float32),responsible_link=np.asarray(link_name),body_region=np.asarray(body_region),sensor_visibility=np.asarray(visibility))
    result={"schema":"h1_articulated_swept_geometry_sufficiency_v1_result","experiment":"H1_ARTICULATED_SWEPT_GEOMETRY_SUFFICIENCY_V1","source_commit":"ea361860afbbd814fd7110a5b8ea504ff83293b9",
      "claim_boundary":"SIMULATED_DISALLOWED_CONTACT_PROXY over the committed H1 block only; no material-hazard, injury, property, human, fragile-infrastructure, or closed-loop guarantee",
      "control_horizon":{"blocks_committed":1,"ticks":5,"command_tick_s":.1,"duration_s":.5,"replanning":"immediately after H1","blocks_2_to_4_replaceable":True,"hold_next_cycle":True,"validated_emergency_brake":False,"H2_stopping_status":"NOT_A_VALIDATED_STOPPING_HORIZON"},
      "bindings":{"panel_digest":panel["content_digest"],"geometry_index_digest":index["content_digest"],"wide_geometry_checkpoint_identity_only":"3e556531a0442df214d0667ad42110e42806ec3aa7aa240c2b2746d7c304af31","checkpoint_executed":False},
      "materialisation":{"replayed_states":index["replayed_states"],"replayed_branches":index["replayed_branches"],"physics_steps":index["physics_steps_total"],"verification_failures":index["verification_failures"],"new_identities":0,"contracts":index["contracts"],"runtime_compute_s":index["runtime_compute_s"],"parallel_wall_runtime_s":index["parallel_wall_runtime_s"],"storage_bytes":index["storage_bytes"]},
      "fixture":fixture,"calibration":calibration,"heldout":heldout,"heldout_frontiers":frontiers,"replay_mismatch_sensitivity":sensitivity,"physics_step_label_comparison":physics_comparison(held_rows,evidence),"contact_attribution":attr,
      "classification":classification,"secondary_classifications":secondary,
      "next_step":("ARTICULATED_CONTACT_DYNAMICS_STATE_V1" if classification=="FULL_ARTICULATED_GEOMETRY_CONTACT_PROXY_NO_GO" else
          "CANDIDATE_CONDITIONED_H1_GEOMETRY_PREDICTOR_V1" if classification.endswith("SWEPT_GEOMETRY_SIGNAL") else "denser vertical LiDAR/body-facing sensing or narrower contact-avoidance claim"),
      "row_level_evidence":{"path":str(evidence_path),"sha256":sha(evidence_path),"bytes":evidence_path.stat().st_size},"runtime":{"evaluation_s":time.time()-started},
      "prohibitions_confirmed":{"model_training":0,"learned_checkpoint_inference":0,"jepa_access":0,"new_state_identities":0,"memory_navigation_novelty":0}}
    result["content_digest"]=GEO.digest(result); atomic_json(OUT/"result.json",result)
    print(json.dumps({"classification":classification,"secondary":secondary,"thresholds":thresholds,"full_gate":heldout[CONDITIONS[0]]["gate"],"row_ledger_sha256":result["row_level_evidence"]["sha256"]},indent=2)); return 0

if __name__=="__main__": raise SystemExit(main())
