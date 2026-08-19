#!/usr/bin/env python3
from __future__ import annotations
import hashlib,json,random,time
from pathlib import Path
import numpy as np, torch
import torch.nn as nn
import torch.nn.functional as F
from dev_frozen_dense_representation_encoders_v1 import VJepa21Arm
ROOT=Path(__file__).resolve().parents[1]; INV=ROOT/'.generated/go2_memory_role_place_triplet_index_v1/train.jsonl'; OUT=Path('/home/andrewknowles/.cache/lewm_go2_temporal_v03/place_head_dev_v2'); SEED=2026081902
class Head(nn.Module):
 def __init__(self):
  super().__init__(); self.m=nn.Sequential(nn.LayerNorm(1024),nn.Linear(1024,256),nn.GELU(),nn.Linear(256,128)); self.o=nn.Linear(256,128)
 def forward(self,x):
  z=self.m(x); return F.normalize(self.o(torch.cat([z.mean(1),z.amax(1)],1)),dim=-1)
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def load():
 rows=[json.loads(x) for x in open(INV)]; split=json.load(open(OUT/'split.json')); idx=json.load(open(OUT/'cache_index.json')); cache={p:torch.from_numpy(np.load(q).copy()).float() for p,q in idx.items()}; return rows,split,cache
def desc(model,x,dev): return model(x.unsqueeze(0).to(dev)).squeeze(0).detach().cpu()
def retrieval(model,rows,scenes,cache,dev):
 rows=[r for r in rows if r['scene_id'] in scenes]; out=[]
 for r in rows:
  q=desc(model,cache[r['anchor']['rgb_path']],dev); target=r['positive']['endpoint_identity_sha256']; gal={}
  for z in rows:
   if z['scene_id']!=r['scene_id']: continue
   gal.setdefault(z['positive']['endpoint_identity_sha256'],[]).append(desc(model,cache[z['positive']['rgb_path']],dev))
  scores=sorted([(float((torch.stack(v)@q).max()),k) for k,v in gal.items()],reverse=True); rank=next((i+1 for i,(_,k) in enumerate(scores) if k==target),len(scores)); wrong=[s for s,k in scores if k!=target]; out.append({'rank':rank,'top1':rank==1,'top3':rank<=3,'family':r['family'],'margin':scores[0][0]-(wrong[0] if wrong else scores[0][0])})
 return out
def summary(v,nested=True):
 return {'queries':len(v),'top1':sum(x['top1'] for x in v)/len(v),'top3':sum(x['top3'] for x in v)/len(v),'mrr':sum(1/x['rank'] for x in v)/len(v),'median_rank':float(np.median([x['rank'] for x in v])),'mean_margin':sum(x['margin'] for x in v)/len(v),'per_family':{f:summary([x for x in v if x['family']==f],False) for f in sorted({x['family'] for x in v})} if nested else {}}
def main():
 random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED); rows,split,cache=load(); dev=torch.device('cuda' if torch.cuda.is_available() else 'cpu'); fit_s=set(split['fit_scenes']); ev_s=set(split['eval_scenes']); fit=[r for r in rows if r['scene_id'] in fit_s]; ev=[r for r in rows if r['scene_id'] in ev_s]
 # V2 attribution: registered triplet margins and fit/held-out retrieval.
 v2=Head().to(dev); v2.load_state_dict(torch.load(OUT/'place_head_dev_v2.pt',map_location=dev)['state_dict']); v2.eval()
 margins=[]; pos=[]; neg=[]; comp=[]
 with torch.no_grad():
  for r in rows:
   a,p,n=[desc(v2,cache[r[k]['rgb_path']],dev) for k in ('anchor','positive','negative')]; pc=float(a@p); nc=float(a@n); margins.append(.2+nc-pc); pos.append(pc); neg.append(nc)
 attr={'fit_retrieval':summary(retrieval(v2,fit,fit_s,cache,dev)),'heldout_retrieval':summary(retrieval(v2,ev,ev_s,cache,dev)),'fit_triplet_margin_satisfaction':sum(x<=0 for x in margins[:len(fit)])/len(fit),'heldout_triplet_margin_satisfaction':sum(x<=0 for x in margins[len(fit):])/len(margins[len(fit):]),'positive_cosine_mean':float(np.mean(pos)),'negative_cosine_mean':float(np.mean(neg)),'descriptor_dim_variance_mean':None,'registered_negative_composition':{'same_scene':None,'different_scene':None,'same_family':None,'different_family':None}}
 # Build fixed scene/node/view inventory.
 inv={}
 for r in fit:
  s=r['scene_id']; n=r['positive']['endpoint_identity_sha256']; inv.setdefault(s,{}).setdefault(n,set()).update([r['anchor']['rgb_path'],r['positive']['rgb_path']])
 scenes=sorted([s for s,n in inv.items() if len(n)>=4 and sum(len(v)>=2 for v in n.values())>=4]); rng=random.Random(SEED); batches=[]
 for _ in range(max(1,len(scenes)*2)):
  ss=rng.sample(scenes,min(4,len(scenes))); samples=[]
  for s in ss:
   ns=[n for n,v in inv[s].items() if len(v)>=2]; ns=rng.sample(ns,min(4,len(ns)))
   for n in ns:
    vv=rng.sample(sorted(inv[s][n]),2); samples.extend(vv)
  if len(samples)==len(ss)*8: batches.append(samples)
 model=Head().to(dev); opt=torch.optim.AdamW(model.parameters(),lr=1e-3,weight_decay=1e-4); losses=[]; t=time.time()
 for ep in range(30):
  rng.shuffle(batches); vals=[]
  for samples in batches:
   x=torch.stack([cache[p] for p in samples]).to(dev); z=model(x); # 8 samples/scene, paired views adjacent by node
   sim=z@z.T/.07; mask=torch.eye(len(z),device=dev,dtype=torch.bool); sim=sim.masked_fill(mask,-1e9); pos_idx=torch.tensor([i^1 for i in range(len(z))],device=dev); loss=-(sim[torch.arange(len(z),device=dev),pos_idx]-torch.logsumexp(sim,1)).mean(); opt.zero_grad(); loss.backward(); opt.step(); vals.append(float(loss))
  losses.append(float(np.mean(vals))); print('epoch',ep+1,losses[-1],flush=True)
 ck=OUT/'place_head_dev_v3.pt'; torch.save({'state_dict':model.state_dict(),'seed':SEED,'losses':losses},ck); result={'schema':'place_head_within_scene_supcon_v3','seed':SEED,'fit_scenes':len(fit_s),'eval_scenes':len(ev_s),'fit_rows':len(fit),'eval_rows':len(ev),'sampling_batches_per_epoch':len(batches),'failure_attribution':attr,'losses':losses,'checkpoint_sha256':sha(ck),'training_seconds':time.time()-t,'v3_heldout':summary(retrieval(model,ev,ev_s,cache,dev)),'v2_heldout':attr['heldout_retrieval']}; json.dump(result,open(OUT/'v3_result.json','w'),sort_keys=True,indent=2); print(json.dumps(result,indent=2))
if __name__=='__main__': main()
