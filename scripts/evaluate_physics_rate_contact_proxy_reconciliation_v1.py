#!/usr/bin/env python3
"""Read-only reconciliation of sampled and 2 ms H1 contact targets."""
from __future__ import annotations

from collections import Counter, defaultdict
import hashlib
import json
import math
import os
from pathlib import Path
import time

import numpy as np

from lewm.safety import articulated_swept_geometry_v1 as GEO
from scripts import evaluate_h1_articulated_swept_geometry_sufficiency_v1 as PREV
from scripts import evaluate_wide_geometry_score_composition_v1 as BASE

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/".generated/physics_rate_contact_proxy_reconciliation_v1"
CACHE=Path("/home/andrewknowles/.cache/lewm_go2_temporal_v03/physics_rate_contact_proxy_reconciliation_v1")
GEOMETRY_INDEX=ROOT/".generated/h1_articulated_swept_geometry_sufficiency_v1/articulated_geometry_index.json"
GEOMETRY_RESULT=ROOT/".generated/h1_articulated_swept_geometry_sufficiency_v1/result.json"
GEOMETRY_LEDGER=Path("/home/andrewknowles/.cache/lewm_go2_temporal_v03/h1_articulated_swept_geometry_sufficiency_v1/row_level_evidence_v1.npz")
GEOMETRY_LEDGER_SHA="827263fa58aaf782daddcca9c935173f46a0b4c44a672549cbc2daf8b4a7eea5"
WIDE_LEDGER=Path("/home/andrewknowles/.cache/lewm_go2_temporal_v03/wide_geometry_embodied_contact_proxy_v1/stage1/row_level_evidence_v1.npz")
OLD_LEARNED_ROOT=Path("/home/andrewknowles/.cache/lewm_go2_temporal_v03/geometry_modality_safety_sufficiency_v1")
CONDITIONS=PREV.CONDITIONS
FAMILIES=BASE.FAMILIES


def sha(path: Path) -> str:
    h=hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda:f.read(1<<22),b""): h.update(b)
    return h.hexdigest()


def atomic_json(path: Path,value) -> None:
    path.parent.mkdir(parents=True,exist_ok=True); tmp=path.with_name(f".{path.name}.tmp-{os.getpid()}")
    tmp.write_text(json.dumps(value,indent=2,sort_keys=True,allow_nan=False)+"\n"); os.replace(tmp,path)


def atomic_npz(path: Path,**arrays) -> None:
    path.parent.mkdir(parents=True,exist_ok=True); tmp=path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with tmp.open("wb") as f: np.savez_compressed(f,**arrays)
    os.replace(tmp,path)


def summarize(values) -> dict:
    x=np.asarray(values,np.float64)
    if not len(x): return {"count":0}
    return {"count":int(len(x)),"min":float(x.min()),"q25":float(np.quantile(x,.25)),"median":float(np.median(x)),
      "mean":float(x.mean()),"q75":float(np.quantile(x,.75)),"max":float(x.max())}


def events(trace: np.ndarray) -> list[tuple[int,int]]:
    padded=np.pad(np.asarray(trace,bool).astype(np.int8),(1,1)); starts=np.flatnonzero(np.diff(padded)==1); ends=np.flatnonzero(np.diff(padded)==-1)
    return [(int(a),int(b-1)) for a,b in zip(starts,ends,strict=True)]


def body_region(link: str) -> str:
    if link=="base": return "trunk"
    if link.startswith(("FL","FR")): return "front_limb"
    if link.startswith(("RL","RR")): return "rear_limb"
    return "unresolved"


def load_evidence():
    if sha(GEOMETRY_LEDGER)!=GEOMETRY_LEDGER_SHA: raise RuntimeError("geometry ledger SHA mismatch")
    index=json.loads(GEOMETRY_INDEX.read_text()); predecessor=json.loads(GEOMETRY_RESULT.read_text())
    with np.load(GEOMETRY_LEDGER,allow_pickle=False) as f: old={k:np.asarray(f[k]) for k in f.files}
    old_map={str(b):i for i,b in enumerate(old["branch_id"])}; rows=[]; traces={}; contracts={}
    for state in index["state_records"]:
        contracts[state["state_id"]]=state
        with np.load(state["shard_path"],allow_pickle=False) as f:
            for ci in range(12):
                bid=f"{state['state_id']}:{ci:02d}"; oi=old_map[bid]
                trace=np.asarray(f["physics_contact"][ci],bool); clear=np.asarray(f["clearance"][ci],np.float64); arg=np.asarray(f["arg_geom"][ci]); obj=np.asarray(f["arg_object"][ci])
                spans=events(trace); first=None if not spans else spans[0][0]; final=None if not spans else spans[-1][1]
                sampled=bool(old["h1_contact"][oi]); physics=bool(trace.any())
                nearest=[]
                for start,_end in spans:
                    gi=int(arg[start,0]); nearest.append(state["collision_shape_contract"][gi]["link_name"] if gi>=0 else "unresolved")
                rows.append({"branch_id":bid,"state_id":str(old["state_id"][oi]),"candidate_index":int(old["candidate_index"][oi]),"family":str(old["family"][oi]),"split":str(old["split"][oi]),
                  "sampled_contact":sampled,"hard_contact":physics,"physics_positive_steps":int(trace.sum()),"first_physics_step":first,"final_physics_step":final,"event_count":len(spans),
                  "duration_s":float(trace.sum()*.002),"event_durations_steps":[b-a+1 for a,b in spans],"only_between_samples":bool(physics and not sampled),"visible_at_sample":sampled,
                  "nearest_links":nearest,"body_regions":sorted(set(body_region(x) for x in nearest)),"stuck":bool(old["stuck"][oi]),"p_d":float(old["p_d"][oi]),"p_theta":float(old["p_theta"][oi]),
                  "kinematic":np.asarray(old["kinematic"][oi],np.float64),"scores":np.asarray(old["score"][oi],np.float64)})
                traces[bid]={"contact":trace,"clearance":clear,"arg_geom":arg,"arg_object":obj}
    if len(rows)!=576 or len({r["branch_id"] for r in rows})!=576: raise RuntimeError("row cardinality/alignment failure")
    return index,predecessor,old,rows,traces,contracts


def rows_for(rows,split,score_index=None,score_values=None):
    chosen=[r for r in rows if r["split"]==split]
    if score_values is not None:
        return [{**r,"score_scalar":float(score_values[r["branch_id"]])} for r in chosen]
    return [{**r,"score_scalar":float(r["scores"][score_index])} for r in chosen]


def evaluate(rows,threshold,*,geometry=True):
    states=BASE.prepare_states(rows); scores=np.asarray([r["score_scalar"] for r in rows]); admitted=scores < (-threshold if geometry else threshold)
    metric=BASE.decision_metrics(rows,states,admitted,True); labels=np.asarray([r["hard_contact"] for r in rows],bool)
    metric.update({"auc":BASE.auc(labels,scores),"average_precision":BASE.average_precision(labels,scores),"threshold":float(threshold),
      "per_family":BASE.family_metrics(rows,states,admitted)})
    return metric,admitted


def select_threshold(rows,minimum_recall,*,geometry=True):
    scores=np.asarray([r["score_scalar"] for r in rows]); base=-scores if geometry else scores
    thresholds=np.concatenate(([np.nextafter(base.min(),-np.inf)],np.unique(base),[np.nextafter(base.max(),np.inf)])); eligible=[]
    for threshold in thresholds:
        metric,_=evaluate(rows,float(threshold),geometry=geometry)
        if metric["contact_recall"]+1e-15 < minimum_recall: continue
        key=(metric["states_retaining_contact_negative"],metric["contact_negative_retention"],-metric["selected_contact_count"],metric["mean_selected_route_progress_m"],
          -(metric["normalized_route_progress_regret"] if metric["normalized_route_progress_regret"] is not None else 1e9),-metric["false_abstentions"],float(threshold) if geometry else -float(threshold))
        eligible.append((key,float(threshold),metric))
    if not eligible: raise RuntimeError("no eligible threshold")
    selected=max(eligible,key=lambda x:x[0]); return selected[1],{"threshold_count":len(thresholds),"eligible_count":len(eligible),"metrics":selected[2]}


def full_gate(metric):
    family_ok=all(v["states_retaining_contact_negative"]>0 and v["selected_contact_count"]==0 for v in metric["per_family"].values())
    checks={"auc_ge_0_99":metric["auc"]>=.99,"ap_ge_0_99":metric["average_precision"]>=.99,"recall_ge_0_99":metric["contact_recall"]>=.99,
      "fnr_le_0_01":metric["contact_false_negative_rate"]<=.01,"retention_ge_0_90":metric["contact_negative_retention"]>=.90,
      "states_ge_22":metric["states_retaining_contact_negative"]>=22,"no_only_positive":metric["states_only_contact_positive_admitted"]==0,
      "zero_selected_contact":metric["selected_contact_count"]==0,"false_abstentions_le_1":metric["false_abstentions"]<=1,
      "progress_fraction_ge_0_95":metric["oracle_progress_fraction"] is not None and metric["oracle_progress_fraction"]>=.95,
      "regret_le_0_10":metric["normalized_route_progress_regret"] is not None and metric["normalized_route_progress_regret"]<=.10,
      "top3_ge_0_90":metric["best_contact_negative_top3"] is not None and metric["best_contact_negative_top3"]>=.90,"no_family_collapse":family_ok}
    return {"passed":all(checks.values()),"checks":checks}


def sensor_gate(metric):
    family_ok=all(v["states_retaining_contact_negative"]>0 and v["selected_contact_count"]==0 for v in metric["per_family"].values())
    checks={"auc_ge_0_90":metric["auc"]>=.90,"ap_ge_0_90":metric["average_precision"]>=.90,"recall_ge_0_95":metric["contact_recall"]>=.95,
      "fnr_le_0_05":metric["contact_false_negative_rate"]<=.05,"retention_ge_0_50":metric["contact_negative_retention"]>=.50,
      "states_ge_18":metric["states_retaining_contact_negative"]>=18,"no_only_positive":metric["states_only_contact_positive_admitted"]==0,
      "zero_selected_contact":metric["selected_contact_count"]==0,"false_abstentions_le_3":metric["false_abstentions"]<=3,
      "progress_fraction_ge_0_80":metric["oracle_progress_fraction"] is not None and metric["oracle_progress_fraction"]>=.80,
      "regret_le_0_20":metric["normalized_route_progress_regret"] is not None and metric["normalized_route_progress_regret"]<=.20,
      "top3_ge_0_75":metric["best_contact_negative_top3"] is not None and metric["best_contact_negative_top3"]>=.75,"no_family_collapse":family_ok}
    return {"passed":all(checks.values()),"checks":checks}


def frontier(rows,condition,*,geometry=True,learned=False):
    scores=np.asarray([r["score_scalar"] for r in rows]); base=-scores if geometry else scores
    thresholds=np.concatenate(([np.nextafter(base.min(),-np.inf)],np.unique(base),[np.nextafter(base.max(),np.inf)])); arrays=defaultdict(list); passes=0
    for threshold in thresholds:
        metric,_=evaluate(rows,float(threshold),geometry=geometry); gate=sensor_gate(metric) if condition!=CONDITIONS[0] else full_gate(metric)
        if learned: gate=sensor_gate(metric)
        for key,value in {"threshold":threshold,"recall":metric["contact_recall"],"retention":metric["contact_negative_retention"],"states":metric["states_retaining_contact_negative"],
          "selected_contacts":metric["selected_contact_count"],"progress":metric["mean_selected_route_progress_m"],"regret":metric["normalized_route_progress_regret"],"gate":gate["passed"]}.items(): arrays[key].append(np.nan if value is None else value)
        passes+=gate["passed"]
    data={k:np.asarray(v) for k,v in arrays.items()}; name=condition.lower(); path=CACHE/"frontiers"/f"{name}.npz"; atomic_npz(path,**data)
    safe=np.asarray(data["recall"])>=.95; zero=np.asarray(data["selected_contacts"])==0
    def best(key,mask,maximum=True):
        v=np.asarray(data[key],float)[mask&np.isfinite(data[key])]; return None if not len(v) else float(v.max() if maximum else v.min())
    return {"thresholds":len(thresholds),"complete_gate_points":int(passes),"any_complete_gate":bool(passes),"maximum_negative_retention_at_recall_ge_0_95":best("retention",safe),
      "maximum_states_retaining_at_recall_ge_0_95":best("states",safe),"maximum_progress_with_zero_selected_contact":best("progress",zero),
      "minimum_regret_at_recall_ge_0_95_and_zero_selected":best("regret",safe&zero,False),"path":str(path),"sha256":sha(path)}


def confusion_and_aliasing(rows):
    result={}
    for split in ("calibration","heldout"):
        selected=[r for r in rows if r["split"]==split]; sampled=np.asarray([r["sampled_contact"] for r in selected]); physics=np.asarray([r["hard_contact"] for r in selected])
        tp=int((sampled&physics).sum()); fp=int((sampled&~physics).sum()); fn=int((~sampled&physics).sum()); tn=int((~sampled&~physics).sum())
        missed=[r for r in selected if r["only_between_samples"]]; durations=[d for r in missed for d in r["event_durations_steps"]]
        by_family={f:{"branches":sum(r["family"]==f for r in selected),"sampled_positive":sum(r["family"]==f and r["sampled_contact"] for r in selected),
          "physics_positive":sum(r["family"]==f and r["hard_contact"] for r in selected),"missed":sum(r["family"]==f and r["only_between_samples"] for r in selected)} for f in FAMILIES}
        by_candidate={str(c):{"sampled_positive":sum(r["candidate_index"]==c and r["sampled_contact"] for r in selected),"physics_positive":sum(r["candidate_index"]==c and r["hard_contact"] for r in selected),
          "missed":sum(r["candidate_index"]==c and r["only_between_samples"] for r in selected)} for c in range(12)}
        regions=Counter(region for r in missed for region in r["body_regions"])
        result[split]={"branches":len(selected),"confusion":{"sampled_positive_physics_positive":tp,"sampled_positive_physics_negative":fp,
          "sampled_negative_physics_positive":fn,"sampled_negative_physics_negative":tn},"sampled_sensitivity_to_physics_contact":tp/(tp+fn),"sampled_specificity":tn/(tn+fp),
          "physics_contact_missed_fraction":fn/(tp+fn),"sampled_prevalence":float(sampled.mean()),"physics_prevalence":float(physics.mean()),
          "positive_physics_steps":int(sum(r["physics_positive_steps"] for r in selected)),"contact_events":int(sum(r["event_count"] for r in selected)),
          "all_event_duration_steps":summarize([d for r in selected for d in r["event_durations_steps"]]),"missed_event_duration_steps":summarize(durations),
          "missed_one_physics_step_fraction":None if not durations else float(np.mean(np.asarray(durations)==1)),"missed_body_region_nearest_geometry":dict(regions),"by_family":by_family,"by_candidate":by_candidate,
          "impulse_distribution":"UNAVAILABLE_IN_PERSISTED_WIDE_PANEL","relative_speed_distribution":"UNAVAILABLE_IN_PERSISTED_WIDE_PANEL"}
    return result


def fit_temperature(logits,labels):
    logits=np.asarray(logits,np.float64); labels=np.asarray(labels,np.float64)
    def objective(log_t):
        t=min(20.,max(.05,math.exp(log_t))); z=logits/t
        return float(np.mean(np.maximum(z,0)-z*labels+np.log1p(np.exp(-np.abs(z)))))
    lo,hi=math.log(.05),math.log(20.); ratio=(math.sqrt(5)-1)/2; a=hi-ratio*(hi-lo); b=lo+ratio*(hi-lo); fa,fb=objective(a),objective(b)
    for _ in range(240):
        if fa<=fb: hi,b,fb=b,a,fa; a=hi-ratio*(hi-lo); fa=objective(a)
        else: lo,a,fa=a,b,fb; b=lo+ratio*(hi-lo); fb=objective(b)
    return float(math.exp((lo+hi)/2))


def sigmoid(x):
    x=np.asarray(x,np.float64); return np.where(x>=0,1/(1+np.exp(-x)),np.exp(x)/(1+np.exp(x)))


def learned_diagnostics(rows):
    with np.load(WIDE_LEDGER,allow_pickle=False) as f: wide={k:np.asarray(f[k]) for k in f.files}
    row_ids={r["branch_id"] for r in rows}; wide_map={str(b):i for i,b in enumerate(wide["branch_id"])}
    if not row_ids.issubset(wide_map): raise RuntimeError("wide learned ledger lacks current rows")
    raw={bid:float(wide["raw_logits"][wide_map[bid],4,1]) for bid in row_ids}
    cal=[r for r in rows if r["split"]=="calibration"]; temperature=fit_temperature([raw[r["branch_id"]] for r in cal],[r["hard_contact"] for r in cal])
    probability={bid:float(sigmoid(raw[bid]/temperature)) for bid in row_ids}
    cal_rows=rows_for(rows,"calibration",score_values=probability); threshold,selection=select_threshold(cal_rows,.95,geometry=False)
    held_rows=rows_for(rows,"heldout",score_values=probability); metric,admitted=evaluate(held_rows,threshold,geometry=False); gate=sensor_gate(metric); fr=frontier(held_rows,"wide_geometry_embodied_learned",geometry=False,learned=True)
    sampled_negative=np.asarray([not r["sampled_contact"] for r in held_rows]); rejected=~admitted; physics=np.asarray([r["hard_contact"] for r in held_rows])
    denominator=int((sampled_negative&rejected).sum()); explained=int((sampled_negative&rejected&physics).sum()); fraction=None if denominator==0 else explained/denominator
    unavailable={}
    current_ids=np.asarray(sorted(row_ids))
    for name in ("DEPTH_ONLY","LIDAR_ONLY","DEPTH_PLUS_EMBODIED"):
        path=OLD_LEARNED_ROOT/name/"row_level_evidence_v1.npz"
        with np.load(path,allow_pickle=False) as f: ids=np.asarray(f["branch_id"])
        overlap=len(set(ids.tolist())&row_ids)
        unavailable[name]={"status":"UNAVAILABLE_FOR_CURRENT_PANEL","ledger_path":str(path),"row_count":len(ids),"current_panel_overlap":overlap,
          "first_identity":str(ids[0]),"reason":"persisted logits are bound to scale-* states, not the current wide-* panel; checkpoint execution prohibited"}
    return {"WIDE_GEOMETRY_EMBODIED_CONTACT_HEAD_V1":{"status":"EVALUATED_FROM_PERSISTED_LOGITS","temperature":temperature,"threshold":threshold,
      "calibration":selection,"heldout":metric,"gate":gate,"frontier":fr,"sampled_negative_rejections":denominator,"physics_positive_among_them":explained,
      "fraction_previous_false_positives_explained":fraction},"unavailable_conditions":unavailable},probability


def solver_audit(rows,traces,contracts,threshold):
    audit=[]; classes=Counter()
    for row in rows:
        ev=traces[row["branch_id"]]; predicted=bool(row["score_scalar"]>=-threshold)
        if predicted==row["hard_contact"]: continue
        if row["hard_contact"]:
            contact_steps=np.flatnonzero(ev["contact"]); step=int(contact_steps[np.argmin(ev["clearance"][contact_steps,0])])
        else: step=int(np.argmin(ev["clearance"][:,0]))
        clearance=float(ev["clearance"][step,0]); gi=int(ev["arg_geom"][step,0]); oi=int(ev["arg_object"][step]); state=contracts[row["state_id"]]
        geom=state["collision_shape_contract"][gi] if gi>=0 else None; link=None if geom is None else geom["link_name"]
        tolerance=1e-4
        if geom and geom["kind"]=="capsule": tolerance=max(tolerance,float(geom["data"][1])/64)
        if row["hard_contact"] and clearance>0 and clearance<=tolerance: category="NUMERICAL_TOLERANCE"
        elif gi>=0: category="GEOMETRY_QUERY_MISMATCH"
        else: category="UNRESOLVED"
        classes[category]+=1
        audit.append({"branch_id":row["branch_id"],"family":row["family"],"physics_contact":row["hard_contact"],"geometric_prediction":predicted,
          "physics_step":step,"signed_clearance_m":clearance,"robot_link":link,"environment_object":None if oi<0 else state["scene_object_ids"][oi],
          "solver_contact_active":bool(ev["contact"][step]),"collision_margin_m":0.0,"collision_margin_source":"Genesis 0.3.14 narrowphase source uses zero margin for these rigid primitives",
          "penetration_m":"UNAVAILABLE_IN_PERSISTED_WIDE_PANEL","solver_contact_at_positive_geometric_separation":bool(ev["contact"][step] and clearance>0),
          "same_primitive_source":True,"same_query_algorithm":False,"query_limitation":"33-sample capsule axis and analytic/SAT reducer is not Genesis MPR/GJK narrowphase", "classification":category})
    return {"disagreements":len(audit),"classification_counts":dict(classes),"collision_margin_accounting":"zero configured narrowphase margin; no skin offset found",
      "penetration_available":False,"exact_solver_query_reproduced":False,"rows":audit}


def main():
    started=time.time(); OUT.mkdir(parents=True,exist_ok=True); CACHE.mkdir(parents=True,exist_ok=True)
    index,predecessor,old,rows,traces,contracts=load_evidence(); aliasing=confusion_and_aliasing(rows)
    calibration={}; heldout={}; frontiers={}; thresholds={}
    for ci,condition in enumerate(CONDITIONS):
        minimum=.99 if ci==0 else .95; cal=rows_for(rows,"calibration",ci); threshold,selection=select_threshold(cal,minimum,geometry=True); thresholds[condition]=threshold
        cm,_=evaluate(cal,threshold,geometry=True); hm,_=evaluate(rows_for(rows,"heldout",ci),threshold,geometry=True); hm["gate"]=full_gate(hm) if ci==0 else sensor_gate(hm)
        calibration[condition]={"threshold_clearance_m":threshold,"selection":selection,"metrics":cm}; heldout[condition]=hm; frontiers[condition]=frontier(rows_for(rows,"heldout",ci),condition)
    learned,wide_probability=learned_diagnostics(rows)
    full_rows=rows_for(rows,"heldout",0); solver=solver_audit(full_rows,traces,contracts,thresholds[CONDITIONS[0]])
    full_pass=heldout[CONDITIONS[0]]["gate"]["passed"]; passing_sensors=[c for c in CONDITIONS[1:] if heldout[c]["gate"]["passed"]]
    if full_pass and passing_sensors: classification="PHYSICS_RATE_SENSOR_GEOMETRY_SIGNAL"
    elif full_pass: classification="PHYSICS_RATE_GEOMETRY_SIGNAL_SENSOR_COVERAGE_NO_GO"
    else: classification="PHYSICS_RATE_FULL_GEOMETRY_SCORE_NO_GO"
    secondary=[]
    if aliasing["heldout"]["physics_contact_missed_fraction"]>=.20 and full_pass: secondary.append("TEMPORAL_LABEL_ALIASING_CONFIRMED_GEOMETRY_SUFFICIENT")
    mapping={CONDITIONS[1]:"DEPTH_PHYSICS_RATE_CONTACT_SIGNAL",CONDITIONS[2]:"LIDAR_PHYSICS_RATE_CONTACT_SIGNAL",CONDITIONS[3]:"FUSED_PHYSICS_RATE_CONTACT_SIGNAL"}
    secondary.extend(mapping[c] for c in passing_sensors)
    explained=learned["WIDE_GEOMETRY_EMBODIED_CONTACT_HEAD_V1"]["fraction_previous_false_positives_explained"]
    if explained is not None and explained>=.30: secondary.append("PREVIOUS_FALSE_POSITIVES_EXPLAINED_BY_TEMPORAL_ALIASING")
    # Immutable reconciliation ledger. Solver links were not persisted; the
    # link column is explicitly the nearest full-geometry primitive.
    branch_ids=np.asarray([r["branch_id"] for r in rows]); physics=np.asarray([r["hard_contact"] for r in rows],np.uint8); sampled=np.asarray([r["sampled_contact"] for r in rows],np.uint8)
    event_count=np.asarray([r["event_count"] for r in rows],np.int16); first=np.asarray([-1 if r["first_physics_step"] is None else r["first_physics_step"] for r in rows],np.int16)
    final=np.asarray([-1 if r["final_physics_step"] is None else r["final_physics_step"] for r in rows],np.int16); positive_steps=np.asarray([r["physics_positive_steps"] for r in rows],np.int16)
    scores=np.stack([r["scores"] for r in rows]).astype(np.float32); learned_score=np.asarray([wide_probability[r["branch_id"]] for r in rows],np.float32)
    geo_admit=np.stack([scores[:,i] < -thresholds[c] for i,c in enumerate(CONDITIONS)],1).astype(np.uint8)
    wide_threshold=learned["WIDE_GEOMETRY_EMBODIED_CONTACT_HEAD_V1"]["threshold"]; learned_admit=(learned_score<wide_threshold).astype(np.uint8)
    persisted_conditions=list(CONDITIONS)+["WIDE_GEOMETRY_EMBODIED_CONTACT_HEAD_V1"]
    selected_by_state={}
    for condition in CONDITIONS:
        for split_source in (calibration[condition]["metrics"],heldout[condition]):
            for state_metric in split_source["per_state"]:
                selected_by_state[(condition,state_metric["state_id"])]=-1 if state_metric["selected_candidate"] is None else int(state_metric["selected_candidate"])
    learned_sources=(learned["WIDE_GEOMETRY_EMBODIED_CONTACT_HEAD_V1"]["calibration"]["metrics"],learned["WIDE_GEOMETRY_EMBODIED_CONTACT_HEAD_V1"]["heldout"])
    for split_source in learned_sources:
        for state_metric in split_source["per_state"]:
            selected_by_state[("WIDE_GEOMETRY_EMBODIED_CONTACT_HEAD_V1",state_metric["state_id"])]=-1 if state_metric["selected_candidate"] is None else int(state_metric["selected_candidate"])
    selected_candidates=np.asarray([[selected_by_state[(condition,r["state_id"])] for condition in persisted_conditions] for r in rows],np.int16)
    ledger_path=CACHE/"row_level_evidence_v1.npz"; atomic_npz(ledger_path,branch_id=branch_ids,state_id=np.asarray([r["state_id"] for r in rows]),candidate_index=np.asarray([r["candidate_index"] for r in rows],np.int16),
      split=np.asarray([r["split"] for r in rows]),family=np.asarray([r["family"] for r in rows]),sampled_h1_contact=sampled,physics_rate_h1_contact=physics,
      positive_physics_steps=positive_steps,first_contact_step=first,final_contact_step=final,event_count=event_count,duration_s=positive_steps.astype(np.float32)*.002,
      event_duration_steps_json=np.asarray([json.dumps(r["event_durations_steps"],separators=(",",":")) for r in rows]),
      nearest_geometry_links_json=np.asarray([json.dumps(r["nearest_links"],separators=(",",":")) for r in rows]),
      nearest_geometry_body_regions_json=np.asarray([json.dumps(r["body_regions"],separators=(",",":")) for r in rows]),
      geometry_scores=scores,geometry_thresholds=np.asarray([thresholds[c] for c in CONDITIONS]),geometry_admitted=geo_admit,wide_learned_probability=learned_score,
      wide_learned_threshold=np.asarray([wide_threshold]),wide_learned_admitted=learned_admit,p_d=np.asarray([r["p_d"] for r in rows],np.float32),p_theta=np.asarray([r["p_theta"] for r in rows],np.float32),
      kinematic=np.stack([r["kinematic"] for r in rows]).astype(np.float32),condition_names=np.asarray(persisted_conditions),selected_candidate_by_condition=selected_candidates)
    next_step=("freeze H1_ANY_PHYSICS_STEP_DISALLOWED_CONTACT, then prospectively evaluate the passing sensor geometry state on a fresh panel" if full_pass and passing_sensors else
      "freeze H1_ANY_PHYSICS_STEP_DISALLOWED_CONTACT and prospectively validate full geometry before sensor modelling" if full_pass else
      "first implement an exact Genesis-congruent MPR/GJK clearance and contact-manifold persistence audit; ARTICULATED_CONTACT_DYNAMICS_STATE_V1 is justified only if residual mismatches remain")
    result={"schema":"physics_rate_contact_proxy_reconciliation_v1_result","experiment":"PHYSICS_RATE_CONTACT_PROXY_RECONCILIATION_V1","mode":"POST_OUTCOME_DEVELOPMENT_DIAGNOSTIC",
      "source_commit":"5d440911774682f351b8ab7192c89b453226328b","claim_boundary":{"historical":"H1_SAMPLED_DISALLOWED_CONTACT remains authoritative for completed results",
        "development":"H1_ANY_PHYSICS_STEP_DISALLOWED_CONTACT is a simulated separation/contact-avoidance proxy","material_hazard":"SEVERITY_UNRESOLVED"},
      "bindings":{"geometry_ledger_sha256":GEOMETRY_LEDGER_SHA,"geometry_index_content_digest":index["content_digest"],"frozen_states":48,"branches":576,"physics_steps_per_branch":250},
      "aliasing":aliasing,"calibration":calibration,"heldout":heldout,"solver_consistency":solver,"learned_score_diagnostics":learned,"heldout_frontiers":frontiers,
      "classification":classification,"secondary_classifications":secondary,"full_geometry_passed":full_pass,"passing_sensor_conditions":passing_sensors,
      "articulated_contact_dynamics_state_v1_justified":bool(not full_pass and solver["exact_solver_query_reproduced"]),
      "next_step":next_step,"row_level_evidence":{"path":str(ledger_path),"sha256":sha(ledger_path),"bytes":ledger_path.stat().st_size},
      "runtime":{"evaluation_s":time.time()-started,"new_simulation_steps":0,"model_inference":0},"storage":{"new_bytes":ledger_path.stat().st_size+sum(Path(v["path"]).stat().st_size for v in frontiers.values())+Path(learned["WIDE_GEOMETRY_EMBODIED_CONTACT_HEAD_V1"]["frontier"]["path"]).stat().st_size},
      "confirmations":{"training":False,"learned_checkpoint_execution":False,"replay":False,"new_panel_or_identity":False,"jepa_access":False,"memory_navigation_novelty":False}}
    result["content_digest"]=GEO.digest(result); atomic_json(OUT/"result.json",result)
    print(json.dumps({"classification":classification,"secondary":secondary,"confusion":aliasing["heldout"]["confusion"],"full":{k:heldout[CONDITIONS[0]][k] for k in ("auc","average_precision","contact_recall","contact_false_negative_rate","contact_negative_retention","states_retaining_contact_negative","selected_contact_count")},"solver":solver["classification_counts"],"ledger_sha256":result["row_level_evidence"]["sha256"]},indent=2))
    return 0

if __name__=="__main__": raise SystemExit(main())
