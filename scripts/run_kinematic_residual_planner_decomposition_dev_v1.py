#!/usr/bin/env python3
"""Diagnose Route-Intent V2 and conditionally fit one kinematic residual ranker."""
from __future__ import annotations

import hashlib
import json
import math
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts")]
import train_evaluate_safe_local_waypoint_route_intent_v2 as V2

SOURCE = ROOT / ".generated/safe_local_waypoint_route_intent_v2"
V1 = ROOT / ".generated/safe_local_waypoint_purpose_built_v1"
OUT = ROOT / ".generated/kinematic_residual_planner_decomposition_dev_v1"
SEED = 2026082002
DT = 0.10
DELTA_D = 0.03
DELTA_THETA = math.radians(5.0)
EXPECTED = {
    "target_index": "df5e55b6606b0a914603ec99db9f91d1898bfd460e0b83cbd33abb0772da4874",
    "v2_checkpoint": "6ef052a46632bbe400c1eab0bb4c45d4457b160a9492382c2f2297f095db198a",
    "v2_result": "0dd4e3d7d6f10a7693bc51fcb71faf10e9ea89a881c2914787f1fd64c71a83e9",
}


def json_default(value):
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"cannot serialize {type(value).__name__}")


def sha(path: Path) -> str:
    h=hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda:f.read(1<<22),b""):h.update(b)
    return h.hexdigest()


def integrate(post_slew, waypoint):
    x=y=yaw=0.0
    flat=[tick for block in post_slew[:3] for tick in block]
    for vx,vy,wz in flat:
        x += (math.cos(yaw)*float(vx)-math.sin(yaw)*float(vy))*DT
        y += (math.sin(yaw)*float(vx)+math.cos(yaw)*float(vy))*DT
        yaw = V2.wrap(yaw+float(wz)*DT)
    wx,wy,sgoal,cgoal=waypoint; goal_heading=math.atan2(sgoal,cgoal)
    pd=math.hypot(wx,wy)-math.hypot(wx-x,wy-y)
    pt=abs(V2.wrap(goal_heading))-abs(V2.wrap(goal_heading-yaw))
    return np.asarray([x,y,math.sin(yaw),math.cos(yaw),pd,pt],np.float32)


def route_pairs(rows, indices, include_safety=False):
    by=defaultdict(list)
    for i in indices:by[rows[i]["state_id"]].append(i)
    pairs=[]
    for ids in by.values():
        for a,i in enumerate(ids):
            for j in ids[a+1:]:
                if include_safety and rows[i]["unsafe"] != rows[j]["unsafe"]:
                    pref=1 if not rows[i]["unsafe"] else -1
                elif rows[i]["unsafe"] or rows[j]["unsafe"]:
                    continue
                elif abs(rows[i]["p_d"]-rows[j]["p_d"])>DELTA_D:
                    pref=1 if rows[i]["p_d"]>rows[j]["p_d"] else -1
                elif abs(rows[i]["p_theta"]-rows[j]["p_theta"])>DELTA_THETA:
                    pref=1 if rows[i]["p_theta"]>rows[j]["p_theta"] else -1
                else:continue
                pairs.append((i,j,pref))
    return pairs


def route_order(ids, pd,pt):
    remaining=list(ids);ordered=[]
    while remaining:
        best=max(float(pd[i]) for i in remaining)
        near=[i for i in remaining if best-float(pd[i])<=DELTA_D]
        pick=min(near,key=lambda i:(-float(pt[i]),i));ordered.append(pick);remaining.remove(pick)
    return ordered


def substitution_eval(rows, indices, safety_mask, pd,pt,name, *, rank_score=None, abstain_pd=None, abstain_pt=None):
    by=defaultdict(list)
    for i in indices:by[rows[i]["state_id"]].append(i)
    per=[];norm=[];absreg=[];headreg=[];top1=[];top3=[];selected_safe=[];selected_pd=[];selected_pt=[];abst=0
    for sid,ids in sorted(by.items(),key=lambda x:int(x[0].split('-')[1])):
        admiss=[i for i in ids if bool(safety_mask[i])]
        ranked=(sorted(admiss,key=lambda i:(-float(rank_score[i]),i)) if rank_score is not None else route_order(admiss,pd,pt)) if admiss else []
        gate_pd=pd if abstain_pd is None else abstain_pd;gate_pt=pt if abstain_pt is None else abstain_pt
        improving=[i for i in ranked if gate_pd[i]>DELTA_D or gate_pt[i]>DELTA_THETA]
        pick=improving[0] if improving else None
        safe=[i for i in ids if not rows[i]["unsafe"]];best=V2.best_safe([rows[i] for i in ids])
        best_i=None if best is None else next(i for i in ids if rows[i] is best)
        if pick is None:abst+=1
        else:
            selected_safe.append(not rows[pick]["unsafe"]);selected_pd.append(rows[pick]["p_d"]);selected_pt.append(rows[pick]["p_theta"])
        if best_i is not None:
            top1.append(pick==best_i);top3.append(best_i in ranked[:3])
            if pick is not None and not rows[pick]["unsafe"]:
                vals=[rows[i]["p_d"] for i in safe];r=rows[best_i]["p_d"]-rows[pick]["p_d"];absreg.append(r)
                if len(vals)>=2 and max(vals)-min(vals)>1e-8:norm.append(r/(max(vals)-min(vals)))
                if V2.state_class([rows[i] for i in ids])=="ALIGNMENT_PROGRESS_AVAILABLE":headreg.append(rows[best_i]["p_theta"]-rows[pick]["p_theta"])
        per.append({"state_id":sid,"family":rows[ids[0]]["family"],"class":V2.state_class([rows[i] for i in ids]),
                    "selected_candidate":None if pick is None else rows[pick]["candidate_index"],
                    "selected_safe":None if pick is None else not rows[pick]["unsafe"],
                    "selected_p_d":None if pick is None else rows[pick]["p_d"],
                    "selected_p_theta_deg":None if pick is None else math.degrees(rows[pick]["p_theta"]),
                    "best_safe_candidate":None if best_i is None else rows[best_i]["candidate_index"]})
    fam={}
    for f in V2.FAMILIES:
        q=[x for x in per if x["family"]==f];m=[x for x in q if x["selected_candidate"] is not None]
        fam[f]={"states":len(q),"abstentions":len(q)-len(m),"safe_selection_rate":float(np.mean([x["selected_safe"] for x in m])) if m else None,
                "mean_selected_p_d":float(np.mean([x["selected_p_d"] for x in m])) if m else None,
                "mean_selected_p_theta_deg":float(np.mean([x["selected_p_theta_deg"] for x in m])) if m else None}
    return {"condition":name,"abstention_rate":abst/len(by),"selected_safe_rate":float(np.mean(selected_safe)) if selected_safe else None,
            "selected_unsafe_rate":float(1-np.mean(selected_safe)) if selected_safe else 0.0,
            "mean_selected_p_d":float(np.mean(selected_pd)) if selected_pd else 0.0,
            "mean_selected_p_theta_deg":float(np.degrees(np.mean(selected_pt))) if selected_pt else 0.0,
            "absolute_distance_regret_m":float(np.mean(absreg)) if absreg else None,
            "normalized_safe_progress_regret":float(np.mean(norm)) if norm else None,
            "heading_regret_deg":float(np.degrees(np.mean(headreg))) if headreg else None,
            "best_safe_top1":float(np.mean(top1)) if top1 else None,"best_safe_top3":float(np.mean(top3)) if top3 else None,
            "per_state":per,"per_family":fam}


def component_metrics(rows,pred,indices,cal,support,name):
    base=V2.evaluate(rows,pred,indices,cal,support,name)
    y=np.asarray([rows[i]["unsafe"] for i in indices],bool);p=V2.probs(pred[indices,6],cal)
    base["safety"]["unsafe_candidate_admission"] = float(np.mean(p[y] < cal["unsafe_threshold"])) if y.any() else None
    base["decision"]={"selected_safe_rate":1-base["safety"]["selected_unsafe_rate"],
                      "selected_distance_progress":base["task"]["mean_selected_p_d"],
                      "selected_heading_improvement_deg":base["task"]["mean_selected_p_theta_deg"],
                      "correct_abstention":base["abstention"]["correct_rate"],
                      "false_abstention":base["abstention"]["false_rate_improving_states"]}
    return base


def kinematic_metrics(rows,indices,kin):
    true_xy=np.asarray([rows[i]["motion"][:2] for i in indices]);pred_xy=kin[indices,:2]
    direction=np.sum(true_xy*pred_xy,1)/(np.linalg.norm(true_xy,axis=1)*np.linalg.norm(pred_xy,axis=1)+1e-8)
    true_yaw=np.arctan2([rows[i]["motion"][2] for i in indices],[rows[i]["motion"][3] for i in indices]);pred_yaw=np.arctan2(kin[indices,2],kin[indices,3])
    yaw=np.abs(np.arctan2(np.sin(pred_yaw-true_yaw),np.cos(pred_yaw-true_yaw)))
    pairs=route_pairs(rows,indices);correct=sum(((kin[i,4]-kin[j,4] if abs(rows[i]["p_d"]-rows[j]["p_d"])>DELTA_D else kin[i,5]-kin[j,5])*pref)>0 for i,j,pref in pairs)
    return {"endpoint_error_m":float(np.mean(np.linalg.norm(true_xy-pred_xy,axis=1))),"median_endpoint_error_m":float(np.median(np.linalg.norm(true_xy-pred_xy,axis=1))),
            "direction_cosine":float(np.mean(direction)),"yaw_mae_deg":float(np.degrees(np.mean(yaw))),
            "distance_progress_mae_m":float(np.mean(np.abs(kin[indices,4]-[rows[i]["p_d"] for i in indices]))),
            "heading_progress_mae_deg":float(np.degrees(np.mean(np.abs(kin[indices,5]-[rows[i]["p_theta"] for i in indices])))),
            "route_pairwise_accuracy":float(correct/len(pairs)) if pairs else None,"ordered_pairs":len(pairs)}


class ResidualRanker(nn.Module):
    def __init__(self):
        super().__init__();self.projection=nn.Linear(2070,16,bias=False);self.heads=nn.Linear(16,7,bias=True)
    def forward(self,x):return self.heads(torch.tanh(self.projection(x)))


def residual_predictions(raw,kin):
    out=np.zeros_like(raw);out[:,:2]=kin[:,:2]+raw[:,:2]
    residual_yaw=np.arctan2(raw[:,2],raw[:,3]);kin_yaw=np.arctan2(kin[:,2],kin[:,3]);yaw=kin_yaw+residual_yaw
    out[:,2]=np.sin(yaw);out[:,3]=np.cos(yaw);out[:,4:6]=kin[:,4:6]+raw[:,4:6];out[:,6]=raw[:,6]
    return out


def main():
    OUT.mkdir(parents=True,exist_ok=True);started=time.time()
    bindings={"target_index":sha(SOURCE/"target_latent_index.json"),"v2_checkpoint":sha(SOURCE/"route_intent_planner_seed_2026082001.pt"),"v2_result":sha(SOURCE/"result.json")}
    if bindings!=EXPECTED:raise RuntimeError(f"frozen binding mismatch {bindings}")
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rows,features,_=V2.load_data(device);idx={s:np.asarray([i for i,r in enumerate(rows) if r["split"]==s],int) for s in ("fit","calibration","heldout")}
    checkpoint=torch.load(SOURCE/"route_intent_planner_seed_2026082001.pt",map_location=device,weights_only=False)
    old=V2.RouteIntentRanker().to(device);old.load_state_dict(checkpoint["state_dict"]);old.eval();x=torch.tensor(features,device=device)
    with torch.inference_mode():old_pred=old(x).cpu().numpy();hidden=torch.tanh(old.projection(x)).cpu().numpy()
    old_result=json.loads((SOURCE/"result.json").read_text());cal=old_result["calibration"]
    center=hidden[idx["fit"]].mean(0);support=np.linalg.norm(hidden-center,axis=1)
    split_metrics={s:component_metrics(rows,old_pred,idx[s],cal,support,f"existing_{s}") for s in idx}

    ledger=[json.loads(v) for v in (V1/"branch_labels.jsonl").read_text().splitlines()];by={(r["state_id"],int(r["candidate_index"])):r for r in ledger}
    kin=np.stack([integrate(by[(r["state_id"],r["candidate_index"])]["post_slew"],r["waypoint"]) for r in rows])
    kin_metrics={s:kinematic_metrics(rows,idx[s],kin) for s in idx}
    oracle_safe=np.asarray([not r["unsafe"] for r in rows]);learned_safe=np.zeros(len(rows),bool)
    learned_prob=V2.probs(old_pred[:,6],cal);learned_safe=(learned_prob<cal["unsafe_threshold"])&(support<=cal["support_threshold"])
    true_pd=np.asarray([r["p_d"] for r in rows]);true_pt=np.asarray([r["p_theta"] for r in rows]);ev=idx["heldout"]
    matrix={
      "A_oracle_safety_oracle_route":substitution_eval(rows,ev,oracle_safe,true_pd,true_pt,"A"),
      "B_oracle_safety_action_route":substitution_eval(rows,ev,oracle_safe,kin[:,4],kin[:,5],"B"),
      "C_oracle_safety_learned_route":substitution_eval(rows,ev,oracle_safe,old_pred[:,4],old_pred[:,5],"C"),
      "D_learned_safety_oracle_route":substitution_eval(rows,ev,learned_safe,true_pd,true_pt,"D"),
      "E_learned_safety_learned_route":substitution_eval(rows,ev,learned_safe,old_pred[:,4],old_pred[:,5],"E"),
      "F_runtime_guard_action_route":{"available":False,"reason":"V2 ledger stores no candidate-level planning-time runtime-guard verdict or current obstacle observation; guard requires live local-obstacle inputs."},
      "G_runtime_guard_learned_route":{"available":False,"reason":"same blocker as F"}}
    b,e=matrix["B_oracle_safety_action_route"],matrix["E_learned_safety_learned_route"]
    residual=np.stack([np.asarray([*r["motion"][:2],math.sin(V2.wrap(math.atan2(r["motion"][2],r["motion"][3])-math.atan2(kin[i,2],kin[i,3]))),math.cos(V2.wrap(math.atan2(r["motion"][2],r["motion"][3])-math.atan2(kin[i,2],kin[i,3]))),r["p_d"]-kin[i,4],r["p_theta"]-kin[i,5]],np.float32) for i,r in enumerate(rows)])
    residual_stats={s:{"std":residual[idx[s]].std(0).tolist(),"mean_abs":np.abs(residual[idx[s]]).mean(0).tolist()} for s in idx}
    continuation={"action_oracle_safety_better":b["normalized_safe_progress_regret"] is not None and e["normalized_safe_progress_regret"] is not None and b["normalized_safe_progress_regret"]<=e["normalized_safe_progress_regret"]-.05,
                  "residual_nondegenerate":bool(np.any(residual[idx["fit"]].std(0)>1e-4))}
    bottlenecks=[]
    held=split_metrics["heldout"]
    if held["geometry"]["displacement_direction_cosine"]<.70:bottlenecks.append("ABSOLUTE_MOTION_DECODING_FAILURE")
    if held["route"]["pairwise_accuracy"]<.70:bottlenecks.append("PROGRESS_RANKING_FAILURE")
    if held["safety"]["unsafe_auc"]<.80:bottlenecks.append("SAFETY_DISCRIMINATION_FAILURE")
    if held["safety"]["unsafe_ece"]>.10:bottlenecks.append("CALIBRATION_OR_THRESHOLD_FAILURE")
    if split_metrics["fit"]["route"]["pairwise_accuracy"]-held["route"]["pairwise_accuracy"]>.10:bottlenecks.append("FIT_TO_HELDOUT_GENERALISATION_FAILURE")
    if matrix["D_learned_safety_oracle_route"]["normalized_safe_progress_regret"] is not None and matrix["D_learned_safety_oracle_route"]["normalized_safe_progress_regret"]>.25:bottlenecks.append("CANDIDATE_SELECTION_RULE_FAILURE")

    result={"schema":"kinematic_residual_planner_decomposition_and_dev_v1","status":"POST_OUTCOME_DEVELOPMENT_SUCCESSOR","bindings":bindings,
            "existing_split_decomposition":split_metrics,"component_substitution":matrix,"bottleneck_classification":bottlenecks,
            "kinematic_prior":{"contract":{"dt_s":DT,"horizons":3,"ticks_per_block":5,"integration":"body_twist_Euler_post_slew"},"metrics":kin_metrics,"residual_targets":residual_stats},
            "runtime_safety_guard":{"available":False,"deployable":False,"false_negatives":None,"reason":matrix["F_runtime_guard_action_route"]["reason"]},
            "continuation_gate":continuation,"residual_planner":None,"classification":None,"predictor_evaluation":None}
    if not all(continuation.values()):
        result["classification"]="KINEMATIC_RESIDUAL_CONTINUATION_NO_GO";result["runtime_s"]=time.time()-started
        (OUT/"result.json").write_text(json.dumps(result,indent=2,sort_keys=True,allow_nan=False,default=json_default));print(json.dumps({"classification":result["classification"],"continuation":continuation},default=json_default));return 0

    # One fixed residual seed. No safety output is trained.
    torch.manual_seed(SEED);np.random.seed(SEED)
    residual_input=np.concatenate([features,kin],axis=1).astype(np.float32);rx=torch.tensor(residual_input,device=device);rt=torch.tensor(residual,device=device)
    model=ResidualRanker().to(device);params=sum(p.numel() for p in model.parameters())
    if params>50000:raise RuntimeError(f"residual planner too large: {params}")
    opt=torch.optim.AdamW(model.parameters(),lr=1e-3,weight_decay=1e-4);fit=idx["fit"]
    rank_pairs=[]
    bys=defaultdict(list)
    for i in fit:bys[rows[i]["state_id"]].append(i)
    for ids in bys.values():
        for a,i in enumerate(ids):
            for j in ids[a+1:]:
                if abs(rows[i]["p_d"]-rows[j]["p_d"])>DELTA_D:pref=1 if rows[i]["p_d"]>rows[j]["p_d"] else -1
                elif abs(rows[i]["p_theta"]-rows[j]["p_theta"])>DELTA_THETA:pref=1 if rows[i]["p_theta"]>rows[j]["p_theta"] else -1
                else:continue
                rank_pairs.append((i,j,pref))
    ck=OUT/"kinematic_residual_listwise_ranker_seed_2026082002.pt"
    if ck.exists():
        saved=torch.load(ck,map_location=device,weights_only=False)
        if saved.get("seed") != SEED or saved.get("parameter_count") != params:raise RuntimeError("saved residual checkpoint contract mismatch")
        model.load_state_dict(saved["state_dict"]);history=saved["history"]
    else:
        smoke=model(rx[fit[:16]]);sl=smoke.square().mean();sl.backward()
        if not all(p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum()>0 for p in model.parameters()):raise RuntimeError("residual smoke gradients invalid")
        model.zero_grad();history=[]
        for epoch in range(60):
            raw=model(rx);motion=F.smooth_l1_loss(raw[fit,:4],rt[fit,:4]);progress=F.smooth_l1_loss(raw[fit,4:6],rt[fit,4:6]);ranking=torch.zeros((),device=device)
            for i,j,pref in rank_pairs:ranking+=F.softplus(-pref*(raw[i,6]-raw[j,6]))
            ranking/=max(1,len(rank_pairs));loss=motion+progress+ranking;opt.zero_grad();loss.backward();opt.step()
            history.append({"epoch":epoch+1,"total":float(loss.detach()),"motion":float(motion.detach()),"progress":float(progress.detach()),"ranking":float(ranking.detach())})
            if epoch in (0,9,19,29,39,49,59):print(json.dumps(history[-1]),flush=True)
        torch.save({"state_dict":model.state_dict(),"seed":SEED,"history":history,"parameter_count":params},ck)
    with torch.inference_mode():raw=model(rx).cpu().numpy()
    final=residual_predictions(raw,kin)
    # Selection uses oracle safety upper bound because no deployable guard exists; route score is output 6.
    residual_oracle=substitution_eval(rows,ev,oracle_safe,final[:,4],final[:,5],"residual_oracle_safety",
                                      rank_score=raw[:,6],abstain_pd=final[:,4],abstain_pt=final[:,5])
    kin_oracle=matrix["B_oracle_safety_action_route"];oracle=matrix["A_oracle_safety_oracle_route"]
    geom_res=kinematic_metrics(rows,ev,final);geom_kin=kin_metrics["heldout"]
    pairs=route_pairs(rows,ev);acc=sum(((raw[i,6]-raw[j,6])*pref)>0 for i,j,pref in pairs)/len(pairs)
    residual_oracle["route_pairwise_accuracy"]=float(acc);kin_acc=geom_kin["route_pairwise_accuracy"]
    family_improve=sum((residual_oracle["per_family"][f]["mean_selected_p_d"] or -999)> (kin_oracle["per_family"][f]["mean_selected_p_d"] or -999) for f in V2.FAMILIES)
    checks={"endpoint_error_improves":geom_res["endpoint_error_m"]<geom_kin["endpoint_error_m"],
            "pairwise_improves_0_05":acc>=kin_acc+.05,
            "regret_improves_0_05":residual_oracle["normalized_safe_progress_regret"] is not None and kin_oracle["normalized_safe_progress_regret"] is not None and residual_oracle["normalized_safe_progress_regret"]<=kin_oracle["normalized_safe_progress_regret"]-.05,
            "selected_route_progress_improves":residual_oracle["mean_selected_p_d"]>kin_oracle["mean_selected_p_d"],
            "deployable_safety_no_additional_unsafe":False,
            "three_families_improve":family_improve>=3}
    passed=all(checks.values());classification="KINEMATIC_RESIDUAL_PLANNER_SIGNAL" if passed else "KINEMATIC_RESIDUAL_PLANNER_NO_SIGNAL"
    result["residual_planner"]={"contract":{"name":"KINEMATIC_RESIDUAL_LISTWISE_RANKER_V1","seed":SEED,"parameter_count":params,"input_width":2070,"hidden_width":16,"outputs":["residual_dx","residual_dy","residual_sin_dyaw","residual_cos_dyaw","residual_p_d","residual_p_theta","route_score"],"optimizer":"AdamW","lr":1e-3,"weight_decay":1e-4,"epochs":60},
                                "training":history,"checkpoint":str(ck),"checkpoint_sha256":sha(ck),"heldout_geometry":geom_res,
                                "oracle_safety_result":residual_oracle,"kinematic_baseline":kin_oracle,"oracle_route_upper_bound":oracle,"gate_checks":checks}
    result["classification"]=classification;result["runtime_s"]=time.time()-started;result["predictor_evaluation"]="NOT_REACHED_TRUE_FUTURE_DEVELOPMENT_GATE_FAILED" if not passed else "PENDING"
    path=OUT/"result.json";path.write_text(json.dumps(result,indent=2,sort_keys=True,allow_nan=False,default=json_default))
    print(json.dumps({"classification":classification,"checks":checks,"checkpoint_sha256":sha(ck),"result_sha256":sha(path)},default=json_default))
    return 0


if __name__=="__main__":raise SystemExit(main())
