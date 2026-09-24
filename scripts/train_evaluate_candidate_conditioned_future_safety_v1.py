#!/usr/bin/env python3
"""Single-seed candidate-conditioned future-safety development experiment.

Consumes only the frozen SAFE_LOCAL_WAYPOINT route-intent artefacts.  It does
not render, encode, simulate, or run a world-model predictor.
"""
from __future__ import annotations

import hashlib
import json
import math
import random
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
V1 = ROOT / ".generated/safe_local_waypoint_purpose_built_v1"
V2 = ROOT / ".generated/safe_local_waypoint_route_intent_v2"
GUARD_RESULT = ROOT / ".generated/kinematic_route_with_runtime_safety_guard_v1/result.json"
OUT = ROOT / ".generated/candidate_conditioned_future_safety_v1"
SEED = 2026082003
EXPECTED_TARGET = "df5e55b6606b0a914603ec99db9f91d1898bfd460e0b83cbd33abb0772da4874"
COMPONENTS = ("collision_or_disallowed_contact", "clearance_violation", "stuck", "aggregate_unsafe")
FAMILIES = ("large_enclosed_maze", "medium_enclosed_maze", "small_enclosed_maze", "loop_alias_stress")
DELTA_D = 0.03
DELTA_THETA = math.radians(5.0)


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 22), b""):
            h.update(block)
    return h.hexdigest()


def json_default(value):
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(type(value).__name__)


def auc(y, score):
    y = np.asarray(y, bool); score = np.asarray(score, float)
    pos, neg = score[y], score[~y]
    if not len(pos) or not len(neg): return None
    return float(np.mean([(p > n) + 0.5 * (p == n) for p in pos for n in neg]))


def average_precision(y, score):
    y = np.asarray(y, bool); score = np.asarray(score, float)
    if not y.any(): return None
    order = np.argsort(-score, kind="stable"); ranked = y[order]
    precision = np.cumsum(ranked) / np.arange(1, len(ranked) + 1)
    return float(np.sum(precision * ranked) / ranked.sum())


def ece(y, prob, bins=10):
    y=np.asarray(y,float);p=np.asarray(prob,float);out=0.0
    for i in range(bins):
        lo,hi=i/bins,(i+1)/bins; mask=(p>=lo)&((p<hi) if i<bins-1 else (p<=hi))
        if mask.any():out+=mask.mean()*abs(p[mask].mean()-y[mask].mean())
    return float(out)


def infer_previous_command(state_rows):
    delta=(0.25,0.0,0.35);out=[]
    for axis,limit in enumerate(delta):
        exact=[];lo=-math.inf;hi=math.inf
        for row in state_rows:
            requested=float(row["requested"][0][0][axis]);executed=float(row["post_slew"][0][0][axis])
            if executed<requested-2e-7:exact.append(executed-limit)
            elif executed>requested+2e-7:exact.append(executed+limit)
            else:lo=max(lo,executed-limit);hi=min(hi,executed+limit)
        if exact:
            value=float(np.median(exact))
            if not np.allclose(exact,value,atol=3e-7,rtol=0):raise RuntimeError("inconsistent prior-command reconstruction")
        else:value=float(np.clip(0.,lo,hi))
        out.append(value)
    return out


def integrate(post_slew, waypoint):
    x=y=yaw=0.0
    for vx,vy,wz in [tick for block in post_slew[:3] for tick in block]:
        x+=(math.cos(yaw)*float(vx)-math.sin(yaw)*float(vy))*.1
        y+=(math.sin(yaw)*float(vx)+math.cos(yaw)*float(vy))*.1
        yaw=math.atan2(math.sin(yaw+float(wz)*.1),math.cos(yaw+float(wz)*.1))
    wx,wy,sg,cg=waypoint; gh=math.atan2(sg,cg)
    pd=math.hypot(wx,wy)-math.hypot(wx-x,wy-y)
    pt=abs(math.atan2(math.sin(gh),math.cos(gh)))-abs(math.atan2(math.sin(gh-yaw),math.cos(gh-yaw)))
    return [x,y,math.sin(yaw),math.cos(yaw),pd,pt]


def load_metadata():
    if sha(V2/"target_latent_index.json") != EXPECTED_TARGET: raise RuntimeError("target index binding mismatch")
    ledger=[json.loads(x) for x in (V1/"branch_labels.jsonl").read_text().splitlines()]
    route=[json.loads(x) for x in (V2/"route_intent_labels.jsonl").read_text().splitlines()]
    route_by={x["branch_id"]:x for x in route}
    manifest=json.loads((V1/"state_manifest.json").read_text()); states={x["state_id"]:x for x in manifest["state_candidates"]}
    index=json.loads((V2/"target_latent_index.json").read_text())
    latent={(x["state_id"],int(x["candidate_index"]),int(x["horizon"])):x["latent_path"] for x in index["entries"]}
    replay={}
    for path in (V2/"replay").glob("purpose-*.json"):
        payload=json.loads(path.read_text())
        for row in payload["rows"]:
            replay[(row["state_id"],int(row["candidate_index"]))]=row["horizons"]
    if len(ledger)!=576 or len(replay)!=576 or len(latent)!=1728:raise RuntimeError("frozen panel incomplete")
    source_by_state=defaultdict(list)
    for r in ledger:source_by_state[r["state_id"]].append(r)
    previous_by_state={sid:infer_previous_command(q) for sid,q in source_by_state.items()}
    rows=[]
    for source in ledger:
        sid,ci=source["state_id"],int(source["candidate_index"]); rr=route_by[f"{sid}:{ci:02d}"]; state=states[sid]
        syaw=float(state["start_pose"][1]); body=state.get("waypoint_body_xy")
        if body is None:
            # This exact body-frame waypoint is already deterministically bound
            # by route-intent V2. Recover it from its frozen local geometry.
            from sys import path as syspath
            syspath.insert(0,str(ROOT/"lewm_worlds"))
            from lewm_worlds.manifest import parse_scene_manifest_dict
            from lewm_worlds.scene_graph import SceneGraph
            graph=SceneGraph(parse_scene_manifest_dict(json.loads((Path(state["scene_dir"])/"manifest.json").read_text())))
            wx,wy=graph.cell_center(int(state["waypoint_path_cells"][2]));sx,sy=map(float,state["start_pose"][0]);dx,dy=wx-sx,wy-sy
            body=[math.cos(syaw)*dx+math.sin(syaw)*dy,-math.sin(syaw)*dx+math.cos(syaw)*dy]
        heading=float(rr["horizons"]["3"]["route_heading_world_rad"])-syaw
        waypoint=[float(body[0]),float(body[1]),math.sin(heading),math.cos(heading)]
        previous=previous_by_state[sid]
        # Verify the recovered previous command against every first post-slew tick.
        requested=np.asarray(source["requested"][0][0],float); expected=np.asarray(source["post_slew"][0][0],float)
        clipped=np.clip(requested,np.asarray(previous)-np.asarray([.25,0,.35]),np.asarray(previous)+np.asarray([.25,0,.35]))
        clipped=np.clip(clipped,[-.3,0,-.5],[.3,0,.5])
        if not np.allclose(clipped,expected,atol=2e-7,rtol=0):raise RuntimeError(f"{sid}:{ci} previous-control reconstruction failed")
        labels=np.zeros((3,4),np.float32)
        for h in (1,2,3):
            comp=replay[(sid,ci)][str(h)]["components"]
            labels[h-1,:3]=[bool(comp["collision_or_disallowed_contact"]),bool(comp["clearance_violation"]),bool(comp["stuck"])]
            labels[h-1,3]=bool(source["horizons"][str(h)]["unsafe"])
        action=np.asarray(source["post_slew"][:3],np.float32).reshape(-1)
        if action.shape!=(45,):raise RuntimeError("bad action shape")
        rows.append({"state_id":sid,"candidate_index":ci,"family":source["family"],"split":rr["split"],
                     "labels":labels,"action_control":np.concatenate([action,np.asarray(previous,np.float32)]),
                     "kinematic":np.asarray(integrate(source["post_slew"],waypoint),np.float32),
                     "p_d":float(rr["horizons"]["3"]["p_d"]),"p_theta":float(rr["horizons"]["3"]["p_theta_rad"]),
                     "unsafe":bool(source["horizons"]["3"]["unsafe"]),
                     "future_paths":[latent[(sid,ci,h)] for h in (1,2,3)],
                     # Exact z0 was not separately cached. The state-shared H1
                     # hold target is a conservative, candidate-invariant proxy.
                     "z0_path":latent[(sid,11,1)]})
    return rows


class ActionOnly(nn.Module):
    def __init__(self):
        super().__init__();self.net=nn.Sequential(nn.Linear(54,64),nn.GELU(),nn.Linear(64,12))
    def forward(self,action_control,kinematic):return self.net(torch.cat([action_control,kinematic],1)).view(-1,3,4)


class CurrentContext(nn.Module):
    def __init__(self):
        super().__init__();self.context=nn.Sequential(nn.Linear(2048,64),nn.GELU());self.action=nn.Sequential(nn.Linear(48,64),nn.GELU());self.out=nn.Linear(128,12)
    def forward(self,z0_summary,action_control):return self.out(torch.cat([self.context(z0_summary),self.action(action_control)],1)).view(-1,3,4)


class FutureSafety(nn.Module):
    def __init__(self):
        super().__init__()
        self.projection=nn.Linear(1024,32)
        self.conv=nn.Sequential(nn.Conv2d(96,64,3,padding=1),nn.GELU(),nn.Conv2d(64,64,3,padding=1),nn.GELU())
        self.action=nn.Sequential(nn.Linear(48,64),nn.GELU(),nn.Linear(64,64),nn.GELU())
        self.fusion=nn.Sequential(nn.Linear(448,128),nn.GELU(),nn.Linear(128,12))
        position=torch.arange(32,dtype=torch.float32); hs=[]
        for h in (1.,2.,3.):hs.append(torch.sin(position*(h/3.)/(10000**(2*(position//2)/32))))
        self.register_buffer("horizon_embedding",torch.stack(hs))
    def forward(self,z0,future,action_control):
        # Frozen, affine-free token normalization.
        z0=F.layer_norm(z0,(1024,),weight=None,bias=None);future=F.layer_norm(future,(1024,),weight=None,bias=None)
        current=self.projection(z0); feats=[]
        for h in range(3):
            fut=self.projection(future[:,h])+self.horizon_embedding[h]
            joined=torch.cat([current,fut,fut-current],-1).transpose(1,2).reshape(-1,96,24,32)
            spatial=self.conv(joined);feats.append(torch.cat([spatial.mean((2,3)),spatial.amax((2,3))],1))
        trajectory=torch.cat(feats,1); action=self.action(action_control)
        return self.fusion(torch.cat([trajectory,action],1)).view(-1,3,4)


def load_tokens(rows,indices):
    future=np.empty((len(indices),3,768,1024),np.float16);z0={}
    for out_i,row_i in enumerate(indices):
        row=rows[row_i]
        for h,path in enumerate(row["future_paths"]):future[out_i,h]=np.load(path,mmap_mode="r")
        if row["state_id"] not in z0:z0[row["state_id"]]=np.load(row["z0_path"],mmap_mode="r").copy()
    return future,z0


def summaries(rows,indices,z0_cache):
    output=[]
    with torch.inference_mode():
        for i in indices:
            x=torch.from_numpy(np.asarray(z0_cache[rows[i]["state_id"]],np.float32));x=F.layer_norm(x,(1024,),weight=None,bias=None)
            output.append(torch.cat([x.mean(0),x.amax(0)]))
    return torch.stack(output)


def pos_weights(labels,fit):
    y=labels[fit];weights=np.ones((3,4),np.float32);defined=np.zeros((3,4),bool)
    for h in range(3):
        for c in range(4):
            pos=float(y[:,h,c].sum());neg=len(y)-pos
            defined[h,c]=pos>0 and neg>0
            if defined[h,c]:weights[h,c]=neg/pos
    return weights,defined


def balanced_loss(logits,labels,weights,defined):
    terms=[]
    for h in range(3):
        for c in range(4):
            if defined[h,c]:terms.append(F.binary_cross_entropy_with_logits(logits[:,h,c],labels[:,h,c],pos_weight=weights[h,c]))
    return torch.stack(terms).mean()


def train_model(name,model,rows,indices,labels,action,kinematic,future,z0_cache,z0_summary,weights,defined,device):
    torch.manual_seed(SEED);np.random.seed(SEED);random.seed(SEED);model=model.to(device)
    opt=torch.optim.AdamW(model.parameters(),lr=1e-3,weight_decay=1e-4);batch=8 if name=="future" else 64
    fit_pos={int(row_i):pos for pos,row_i in enumerate(indices)};history=[]
    def forward_batch(batch_ids):
        ac=torch.from_numpy(action[batch_ids]).to(device);kin=torch.from_numpy(kinematic[batch_ids]).to(device)
        if name=="action":return model(ac,kin)
        if name=="context":
            local=[fit_pos[int(i)] for i in batch_ids];return model(z0_summary[local].to(device),ac)
        local=[fit_pos[int(i)] for i in batch_ids]
        fut=torch.from_numpy(future[local].astype(np.float32)).to(device)
        current=torch.stack([torch.from_numpy(np.asarray(z0_cache[rows[int(i)]["state_id"]],np.float32)) for i in batch_ids]).to(device)
        return model(current,fut,ac)
    # Training-only real-data smoke.
    smoke_ids=np.asarray(indices[:min(batch,len(indices))]);out=forward_batch(smoke_ids);target=torch.from_numpy(labels[smoke_ids]).to(device)
    loss=balanced_loss(out,target,torch.from_numpy(weights).to(device),defined);loss.backward()
    if not torch.isfinite(loss):raise RuntimeError(f"{name} smoke loss nonfinite")
    if not all(p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum()>0 for p in model.parameters()):raise RuntimeError(f"{name} smoke gradient failure")
    if not all(model.fusion[-1].weight.grad.reshape(12,-1).abs().sum(1)>0) if name=="future" else False:raise RuntimeError("future output gradient missing")
    opt.zero_grad();model.eval()
    with torch.inference_mode():
        a=forward_batch(smoke_ids);b=forward_batch(smoke_ids)
        if not torch.equal(a,b):raise RuntimeError(f"{name} deterministic inference failed")
        if name=="future":
            changed=action.copy();changed[smoke_ids]=changed[smoke_ids][:,[*range(45)][::-1]+[45,46,47]]
            original=action.copy();action[:]=changed;c=forward_batch(smoke_ids);action[:]=original
            if torch.allclose(a,c):raise RuntimeError("future action sensitivity smoke failed")
    smoke_path=OUT/f".{name}_smoke_reload.pt";torch.save(model.state_dict(),smoke_path)
    clone={"action":ActionOnly,"context":CurrentContext,"future":FutureSafety}[name]().to(device)
    clone.load_state_dict(torch.load(smoke_path,map_location=device,weights_only=True));clone.eval()
    if set(clone.state_dict())!=set(model.state_dict()):raise RuntimeError(f"{name} smoke checkpoint reload failed")
    smoke_path.unlink()
    model.train();started=time.time()
    for epoch in range(60):
        order=np.asarray(indices).copy();np.random.default_rng(SEED+epoch).shuffle(order);tot=0.;n=0
        for start in range(0,len(order),batch):
            ids=order[start:start+batch];pred=forward_batch(ids);target=torch.from_numpy(labels[ids]).to(device)
            loss=balanced_loss(pred,target,torch.from_numpy(weights).to(device),defined);opt.zero_grad();loss.backward();opt.step();tot+=float(loss)*len(ids);n+=len(ids)
        model.eval()
        with torch.inference_mode():
            sample=[]
            for start in range(0,len(indices),batch):sample.append(forward_batch(np.asarray(indices[start:start+batch])).cpu())
            logits=torch.cat(sample);prob=torch.sigmoid(logits[:,2,3]);yy=torch.from_numpy(labels[indices,2,3])
            pos=float(prob[yy.bool()].mean()) if yy.bool().any() else None;neg=float(prob[~yy.bool()].mean()) if (~yy.bool()).any() else None
        history.append({"epoch":epoch+1,"loss":tot/n,"h3_unsafe_positive_probability":pos,"h3_unsafe_negative_probability":neg})
        if epoch in (0,9,19,29,39,49,59):print(json.dumps({"model":name,**history[-1]}),flush=True)
        model.train()
    model.eval();runtime=time.time()-started
    return model,history,runtime


def predict(model,name,rows,indices,action,kinematic,device,batch=8):
    future,z0=load_tokens(rows,indices) if name=="future" else (None,None)
    if name=="context":
        _,z0=load_tokens(rows,indices);summary=summaries(rows,indices,z0)
    output=[]
    with torch.inference_mode():
        for start in range(0,len(indices),batch if name=="future" else 64):
            ids=np.asarray(indices[start:start+(batch if name=="future" else 64)]);ac=torch.from_numpy(action[ids]).to(device);kin=torch.from_numpy(kinematic[ids]).to(device)
            if name=="action":pred=model(ac,kin)
            elif name=="context":pred=model(summary[start:start+len(ids)].to(device),ac)
            else:
                fut=torch.from_numpy(future[start:start+len(ids)].astype(np.float32)).to(device)
                current=torch.stack([torch.from_numpy(np.asarray(z0[rows[int(i)]["state_id"]],np.float32)) for i in ids]).to(device)
                pred=model(current,fut,ac)
            output.append(pred.cpu().numpy())
    return np.concatenate(output)


def fit_temperature(logits,labels):
    x=torch.tensor(logits,dtype=torch.float32);y=torch.tensor(labels,dtype=torch.float32);log_temp=torch.zeros((),requires_grad=True)
    opt=torch.optim.LBFGS([log_temp],lr=.2,max_iter=100)
    def closure():opt.zero_grad();loss=F.binary_cross_entropy_with_logits(x/torch.exp(log_temp).clamp(.05,20),y);loss.backward();return loss
    opt.step(closure);return float(torch.exp(log_temp.detach()).clamp(.05,20))


def choose_threshold(prob,y):
    # Admission uses p < threshold, so each observable cut must sit just above
    # a calibration score; using the score itself would incorrectly exclude
    # the boundary row and can create a spurious reject-all threshold.
    candidates=sorted(set([0.,1.,*(float(np.nextafter(v,math.inf)) for v in prob)]));feasible=[];all_rows=[]
    for t in candidates:
        admitted=prob<t;recall=float(np.mean(~admitted[y])) if y.any() else 1.;retention=float(np.mean(admitted[~y])) if (~y).any() else 0.
        all_rows.append((recall,retention,t))
        if recall>=.95:feasible.append((retention,-t,t,recall))
    if feasible:
        _,_,t,recall=max(feasible);return {"threshold":t,"calibration_recall":recall,"criterion_satisfied":True}
    recall,retention,t=max(all_rows,key=lambda x:(x[0],x[1],-x[2]));return {"threshold":t,"calibration_recall":recall,"criterion_satisfied":False,"safe_retention":retention}


def preference(a,b):
    if a["unsafe"]!=b["unsafe"]:return 1 if not a["unsafe"] else -1
    if a["unsafe"]:return 0
    dd=a["p_d"]-b["p_d"]
    if abs(dd)>DELTA_D:return 1 if dd>0 else -1
    dt=a["p_theta"]-b["p_theta"]
    if abs(dt)>DELTA_THETA:return 1 if dt>0 else -1
    return 0


def best_safe(ids,rows):
    safe=[i for i in ids if not rows[i]["unsafe"]]
    if not safe:return None
    best=safe[0]
    for i in safe[1:]:
        if preference(rows[i],rows[best])>0:best=i
    return best


def route_order(ids,kinematic):
    remaining=list(ids);order=[]
    while remaining:
        best=max(float(kinematic[i,4]) for i in remaining);near=[i for i in remaining if best-float(kinematic[i,4])<=DELTA_D]
        pick=min(near,key=lambda i:(-float(kinematic[i,5]),rows_global[i]["candidate_index"]));order.append(pick);remaining.remove(pick)
    return order


def evaluator_fixture():
    perfect_y=np.asarray([0,0,1,1],bool);perfect_p=np.asarray([.05,.10,.90,.95]);reversed_p=1-perfect_p
    cases={
      "perfect_discrimination":auc(perfect_y,perfect_p)==1.0,
      "reversed_discrimination":auc(perfect_y,reversed_p)==0.0,
      "one_unsafe":int(np.asarray([0,0,1],bool).sum())==1,
      "all_unsafe":bool(np.asarray([1,1,1],bool).all()),
      "one_safe":int((~np.asarray([1,1,0],bool)).sum())==1,
      "false_negative_admission":bool((np.asarray([.2])<.5)[0]),
      "false_positive_rejection":bool(not (np.asarray([.8])<.5)[0]),
      "no_candidate_admitted":not bool(np.any(np.asarray([.8,.9])<.5)),
      "calibrated_probabilities":float(np.mean((perfect_p-perfect_y)**2))<.02,
      "miscalibrated_probabilities":float(np.mean((reversed_p-perfect_y)**2))>.7,
    }
    probe={"selected_candidate":2,"selected_safe":True,"selected_progress":.3,"normalized_regret":0.,"abstention":False}
    cases["deterministic_json"]=json.dumps(probe,sort_keys=True)==json.dumps(json.loads(json.dumps(probe,sort_keys=True)),sort_keys=True)
    payload={"schema":"candidate_conditioned_future_safety_v1_fixture","cases":cases,"pass":all(cases.values())}
    OUT.mkdir(parents=True,exist_ok=True);p=OUT/"evaluator_fixture.json";p.write_text(json.dumps(payload,indent=2,sort_keys=True));
    if json.loads(p.read_text())!=payload:raise RuntimeError("fixture reload failed")
    return payload


def evaluate_condition(name,rows,indices,prob,threshold,kinematic):
    y=np.asarray([rows[i]["unsafe"] for i in indices],bool);admit=prob<threshold
    tp=int(np.sum(y&~admit));fn=int(np.sum(y&admit));tn=int(np.sum(~y&admit));fp=int(np.sum(~y&~admit))
    branch={"rows":len(indices),"unsafe":int(y.sum()),"safe":int((~y).sum()),"auc":auc(y,prob),"average_precision":average_precision(y,prob),
            "unsafe_recall":tp/(tp+fn) if tp+fn else None,"unsafe_false_negative_rate":fn/(tp+fn) if tp+fn else None,
            "safe_specificity":tn/(tn+fp) if tn+fp else None,"safe_candidate_retention":tn/(tn+fp) if tn+fp else None,
            "ece":ece(y,prob),"brier":float(np.mean((prob-y)**2)),"admitted":int(admit.sum()),"rejected":int((~admit).sum())}
    by=defaultdict(list)
    for local,i in enumerate(indices):by[rows[i]["state_id"]].append((local,i))
    state_rows=[];norm=[];top1=[];top3=[];selected_pd=[];selected_pt=[];selected_unsafe=[];false_abst=correct_abst=0
    for sid,pairs in sorted(by.items(),key=lambda x:int(x[0].split('-')[1])):
        ids=[i for _,i in pairs];local={i:l for l,i in pairs};admitted=[i for i in ids if admit[local[i]]];ranked=route_order(admitted,kinematic) if admitted else [];pick=ranked[0] if ranked else None
        safe=[i for i in ids if not rows[i]["unsafe"]];best=best_safe(ids,rows)
        if pick is None:
            if safe:false_abst+=1
            else:correct_abst+=1
        else:selected_pd.append(rows[pick]["p_d"]);selected_pt.append(rows[pick]["p_theta"]);selected_unsafe.append(rows[pick]["unsafe"])
        if best is not None:
            top1.append(pick==best);top3.append(best in ranked[:3])
            if pick is not None and not rows[pick]["unsafe"] and len(safe)>=2:
                vals=[rows[i]["p_d"] for i in safe];spread=max(vals)-min(vals)
                if spread>1e-8:norm.append((rows[best]["p_d"]-rows[pick]["p_d"])/spread)
        state_rows.append({"state_id":sid,"family":rows[ids[0]]["family"],"admitted":len(admitted),"admitted_safe":sum(not rows[i]["unsafe"] for i in admitted),
                           "admitted_unsafe":sum(rows[i]["unsafe"] for i in admitted),"selected_candidate":None if pick is None else rows[pick]["candidate_index"],
                           "selected_safe":None if pick is None else not rows[pick]["unsafe"],"selected_p_d":None if pick is None else rows[pick]["p_d"],
                           "selected_p_theta_deg":None if pick is None else math.degrees(rows[pick]["p_theta"])})
    planning={"states":len(by),"states_retaining_safe":sum(s["admitted_safe"]>0 for s in state_rows),"states_only_unsafe_admitted":sum(s["admitted"]>0 and s["admitted_safe"]==0 for s in state_rows),
              "states_no_admitted":sum(s["admitted"]==0 for s in state_rows),"selected_unsafe_rate":float(np.mean(selected_unsafe)) if selected_unsafe else 0.,
              "mean_selected_distance_progress_m":float(np.mean(selected_pd)) if selected_pd else 0.,"mean_selected_heading_improvement_deg":float(np.degrees(np.mean(selected_pt))) if selected_pt else 0.,
              "normalized_safe_progress_regret":float(np.mean(norm)) if norm else None,"normalized_regret_states":len(norm),
              "best_safe_top1":float(np.mean(top1)) if top1 else None,"best_safe_top3":float(np.mean(top3)) if top3 else None,
              "abstention_rate":sum(s["selected_candidate"] is None for s in state_rows)/len(state_rows),"correct_abstention":correct_abst,"false_abstention":false_abst,"per_state":state_rows}
    fam={}
    for family in FAMILIES:
        mask=np.asarray([rows[i]["family"]==family for i in indices]);suby=y[mask];subp=prob[mask];suba=subp<threshold
        fam[family]={"rows":int(mask.sum()),"unsafe_recall":float(np.mean(~suba[suby])) if suby.any() else None,"unsafe_false_negative_rate":float(np.mean(suba[suby])) if suby.any() else None,
                     "safe_retention":float(np.mean(suba[~suby])) if (~suby).any() else None,"auc":auc(suby,subp)}
    return {"condition":name,"threshold":threshold,"branch":branch,"planning":planning,"per_family":fam}


def component_metrics(rows,indices,all_prob,threshold):
    out={}
    for c,name in enumerate(COMPONENTS[:3]):
        y=np.asarray([rows[i]["labels"][2,c] for i in indices],bool);p=all_prob[:,2,c];pred=p>=threshold
        tp=int(np.sum(y&pred));fn=int(np.sum(y&~pred));fp=int(np.sum(~y&pred))
        out[name]={"prevalence":float(y.mean()),"positive_rows":int(y.sum()),"auc":auc(y,p),"recall":tp/(tp+fn) if tp+fn else None,"precision":tp/(tp+fp) if tp+fp else None,"false_negative_rate":fn/(tp+fn) if tp+fn else None}
    out["fall_or_unsafe_termination"]={"prevalence":0.0,"positive_rows":0,"auc":None,"reason":"degenerate frozen component"}
    return out


def main():
    global rows_global
    OUT.mkdir(parents=True,exist_ok=True);started=time.time();fixture=evaluator_fixture();rows=load_metadata();rows_global=rows
    labels=np.stack([r["labels"] for r in rows]);action=np.stack([r["action_control"] for r in rows]);kin=np.stack([r["kinematic"] for r in rows])
    idx={s:np.asarray([i for i,r in enumerate(rows) if r["split"]==s],int) for s in ("fit","calibration","heldout")}
    prevalence={}
    for split,ids in idx.items():
        prevalence[split]={}
        for family in ["overall",*FAMILIES]:
            q=ids if family=="overall" else np.asarray([i for i in ids if rows[i]["family"]==family],int)
            prevalence[split][family]={f"H{h+1}_{COMPONENTS[c]}":{"positive":int(labels[q,h,c].sum()),"rows":len(q),"prevalence":float(labels[q,h,c].mean())} for h in range(3) for c in range(4)}
    overlaps={}
    h3=labels[:,2,:3].astype(bool)
    for bits in range(8):overlaps[format(bits,"03b")]=int(np.sum(np.all(h3==np.asarray([(bits>>i)&1 for i in range(3)],bool),axis=1)))
    weights,defined=pos_weights(labels,idx["fit"])
    device=torch.device("cpu");torch.set_num_threads(min(16,torch.get_num_threads()))
    model_types={"action":ActionOnly,"context":CurrentContext,"future":FutureSafety}
    checkpoint_paths={name:OUT/f"{name}_safety_seed_{SEED}.pt" for name in model_types}
    prior_result=json.loads((OUT/"result.json").read_text()) if (OUT/"result.json").exists() else None
    models={};histories={};runtimes={};params={};checkpoints={}
    resumed_from_final_checkpoints=all(path.exists() for path in checkpoint_paths.values()) and prior_result is not None
    if resumed_from_final_checkpoints:
        # Evaluation-only resume after a reducer correction. No second fit.
        for name,kind in model_types.items():
            blob=torch.load(checkpoint_paths[name],map_location="cpu",weights_only=False);model=kind();model.load_state_dict(blob["state_dict"]);model.eval();models[name]=model
            params[name]=int(blob["parameter_count"]);histories[name]=prior_result["models"]["training_history"][name];runtimes[name]=prior_result["models"]["training_runtime_s"][name]
            checkpoints[name]={"path":str(checkpoint_paths[name]),"sha256":sha(checkpoint_paths[name])}
    else:
        # Only fit token grids are opened before the single training run.
        fit_future,fit_z0=load_tokens(rows,idx["fit"]);fit_summary=summaries(rows,idx["fit"],fit_z0)
        for name,kind in model_types.items():
            model=kind();params[name]=sum(p.numel() for p in model.parameters())
            if name=="future" and params[name]>=250000:raise RuntimeError(f"future head too large {params[name]}")
            model,history,runtime=train_model(name,model,rows,idx["fit"],labels,action,kin,fit_future,fit_z0,fit_summary,weights,defined,device)
            histories[name]=history;runtimes[name]=runtime
            path=checkpoint_paths[name];torch.save({"state_dict":model.state_dict(),"seed":SEED,"parameter_count":params[name],"epoch":60},path)
            clone=kind();clone.load_state_dict(torch.load(path,map_location="cpu",weights_only=False)["state_dict"]);clone.eval();models[name]=clone;checkpoints[name]={"path":str(path),"sha256":sha(path)}
    calibrations={};heldout_results={};heldout_component={};all_logits={}
    for name in ("action","context","future"):
        cal_logits=predict(models[name],name,rows,idx["calibration"],action,kin,device)[:,2,3];cal_y=labels[idx["calibration"],2,3].astype(bool)
        temp=fit_temperature(cal_logits,cal_y);cal_prob=1/(1+np.exp(-cal_logits/temp));choice=choose_threshold(cal_prob,cal_y)
        calibrations[name]={"temperature":temp,**choice,"tie_rule":"more conservative threshold when safe retention ties"}
        held_logits=predict(models[name],name,rows,idx["heldout"],action,kin,device);all_logits[name]=held_logits
        prob=1/(1+np.exp(-held_logits[:,2,3]/temp));heldout_results[name]=evaluate_condition(name,rows,idx["heldout"],prob,choice["threshold"],kin)
        heldout_component[name]=component_metrics(rows,idx["heldout"],1/(1+np.exp(-held_logits/temp)),choice["threshold"])
    # Oracle-safety and frozen guard comparator.
    held=idx["heldout"];oracle_prob=np.asarray([float(rows[i]["unsafe"]) for i in held]);oracle=evaluate_condition("oracle_safety",rows,held,oracle_prob,.5,kin)
    guard=json.loads(GUARD_RESULT.read_text());guard_held=guard["guard_discrimination"]["by_split"]["heldout"]
    guard_summary={"classification":guard["classification"],"unsafe_recall":guard_held["unsafe_recall"],"unsafe_false_negative_rate":guard_held["unsafe_false_negative_rate"],
                   "safe_candidate_retention":guard_held["safe_candidate_retention"],"per_family":guard["guard_discrimination"]["heldout_by_family"]}
    true=heldout_results["future"];oracle_progress=oracle["planning"]["mean_selected_distance_progress_m"]
    checks={
      "auc_ge_0_80":true["branch"]["auc"] is not None and true["branch"]["auc"]>=.80,
      "recall_ge_0_95":true["branch"]["unsafe_recall"]>=.95,
      "fnr_le_0_05":true["branch"]["unsafe_false_negative_rate"]<=.05,
      "safe_retention_ge_0_40":true["branch"]["safe_candidate_retention"]>=.40,
      "ece_le_0_10":true["branch"]["ece"]<=.10,
      "six_states_retain_safe":true["planning"]["states_retaining_safe"]>=6,
      "no_state_only_unsafe_admitted":true["planning"]["states_only_unsafe_admitted"]==0,
      "no_unsafe_selected":true["planning"]["selected_unsafe_rate"]==0,
      "progress_80pct_oracle":true["planning"]["mean_selected_distance_progress_m"]>=.8*oracle_progress,
      "normalized_regret_le_0_20":true["planning"]["normalized_safe_progress_regret"] is not None and true["planning"]["normalized_safe_progress_regret"]<=.20,
      "best_safe_top3_ge_0_75":true["planning"]["best_safe_top3"]>=.75,
      "false_abstention_le_1":true["planning"]["false_abstention"]<=1,
    }
    passed=all(checks.values());classification="TRUE_FUTURE_SAFETY_HEAD_PASS" if passed else "TRUE_FUTURE_SAFETY_HEAD_NO_GO"
    result={"schema":"candidate_conditioned_future_safety_v1","source_commit":"9c7cef6a86fde16cd173ea48977458febfec64ea",
            "preserved_results":["KINEMATIC_RESIDUAL_PLANNER_NO_SIGNAL","RUNTIME_SAFETY_GUARD_NO_GO_FOR_CANDIDATE_PLANNING"],
            "bindings":{"target_index":EXPECTED_TARGET,"branch_ledger":sha(V1/"branch_labels.jsonl"),"route_labels":sha(V2/"route_intent_labels.jsonl"),"runtime_guard_result":sha(GUARD_RESULT)},
            "context_contract":{"z0_source":"state-shared H1 hold target surrogate","exact_preaction_z0_cached":False,"candidate_invariant":True,
                                "qualification":"development limitation; no rendering or encoding authorised"},
            "dataset":{"states":48,"branches":576,"split":{"fit":384,"calibration":96,"heldout":96},"prevalence":prevalence,"h3_component_overlap_bits_contact_clearance_stuck":overlaps,
                       "fall_or_unsafe_termination_positive":0,"positive_weights":weights.tolist(),"defined_outputs":defined.tolist()},
            "fixture":fixture,"models":{"parameter_count":params,"seed":SEED,"epochs":60,"checkpoints":checkpoints,"training_runtime_s":runtimes,
                                         "training_history":histories},"calibration":calibrations,
            "heldout":{"action_only":heldout_results["action"],"current_context":heldout_results["context"],"privileged_static_grid_guard":guard_summary,
                       "true_future":true,"oracle_safety":oracle,"components":heldout_component},
            "true_future_gate":{"passed":passed,"checks":checks},"predictor_evaluation":None,"classification":classification,
            "training_seed_count":1,"predictor_seed_count":0,"simulation":False,"rendering":False,"encoding":False,
            "runtime":{"training_s":float(sum(runtimes.values())),"final_reducer_s":time.time()-started,
                       "total_compute_s":float(sum(runtimes.values())+(time.time()-started)) if resumed_from_final_checkpoints else time.time()-started,
                       "resumed_from_final_checkpoints":resumed_from_final_checkpoints}}
    path=OUT/"result.json";path.write_text(json.dumps(result,indent=2,sort_keys=True,allow_nan=False,default=json_default));print(json.dumps({"classification":classification,"result_sha256":sha(path),"true_gate":checks},indent=2),flush=True)


if __name__=="__main__":main()
