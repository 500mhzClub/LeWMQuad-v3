#!/usr/bin/env python3
import json,hashlib
from pathlib import Path
import numpy as np, torch
import torch.nn as nn
import torch.nn.functional as F
ROOT=Path(__file__).resolve().parents[1]; OUT=Path('/home/andrewknowles/.cache/lewm_go2_temporal_v03/place_head_dev_v2'); INV=ROOT/'.generated/go2_memory_role_place_triplet_index_v1/train.jsonl'; SEED=2026081901
class Head(nn.Module):
 def __init__(self):
  super().__init__(); self.m=nn.Sequential(nn.LayerNorm(1024),nn.Linear(1024,256),nn.GELU(),nn.Linear(256,128)); self.o=nn.Linear(256,128)
 def forward(self,x):
  z=self.m(x); return F.normalize(self.o(torch.cat([z.mean(1),z.amax(1)],1)),dim=-1)
def main():
 rows=[json.loads(x) for x in open(INV)]; split=json.load(open(OUT/'split.json')); ev=set(split['eval_scenes']); rows=[r for r in rows if r['scene_id'] in ev]; idx=json.load(open(OUT/'cache_index.json')); cache={p:torch.from_numpy(np.load(q).copy()).float() for p,q in idx.items()}; dev=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
 trained=Head().to(dev); trained.load_state_dict(torch.load(OUT/'place_head_dev_v2.pt',map_location=dev)['state_dict']); trained.eval(); torch.manual_seed(SEED); untrained=Head().to(dev).eval()
 def pool(x): return F.normalize(F.layer_norm(x,(1024,)).mean(1),dim=1)
 def run(fn):
  out=[]
  for r in rows:
   q=fn(cache[r['anchor']['rgb_path']].to(dev).unsqueeze(0)).squeeze(0); target=r['positive']['endpoint_identity_sha256']; gallery={};
   for z in rows:
    if z['scene_id']!=r['scene_id']: continue
    k=z['positive']['endpoint_identity_sha256']; gallery.setdefault(k,[]).append(fn(cache[z['positive']['rgb_path']].to(dev).unsqueeze(0)).squeeze(0))
   scores=sorted([(float((torch.stack(v)@q).max()),k) for k,v in gallery.items()],reverse=True); rank=next((i+1 for i,(_,k) in enumerate(scores) if k==target),len(scores)); out.append({'rank':rank,'top1':rank==1,'top3':rank<=3,'family':r['family'],'margin':scores[0][0]-next((s for s,k in scores if k!=target),scores[0][0])})
  return out
 def summary(v, nested=True):
  return {'queries':len(v),'nodes':len({x['family'] for x in v}),'top1':sum(x['top1'] for x in v)/len(v),'top3':sum(x['top3'] for x in v)/len(v),'mrr':sum(1/x['rank'] for x in v)/len(v),'mean_rank':sum(x['rank'] for x in v)/len(v),'median_rank':float(np.median([x['rank'] for x in v])),'mean_margin':sum(x['margin'] for x in v)/len(v),'per_family':{f:summary([x for x in v if x['family']==f],False) for f in sorted({x['family'] for x in v})} if nested else {}}
 result={'trained':summary(run(trained)),'mean_pooled':summary(run(pool)),'untrained':summary(run(untrained)),'checkpoint_sha256':hashlib.sha256(open(OUT/'place_head_dev_v2.pt','rb').read()).hexdigest()}; json.dump(result,open(OUT/'heldout_result.json','w'),sort_keys=True,indent=2); print(json.dumps(result,indent=2))
if __name__=='__main__': main()
