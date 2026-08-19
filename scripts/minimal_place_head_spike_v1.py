#!/usr/bin/env python3
from __future__ import annotations
import hashlib,json,random,time
from pathlib import Path
import numpy as np, torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader,Dataset
from dev_frozen_dense_representation_encoders_v1 import VJepa21Arm
ROOT=Path(__file__).resolve().parents[1]; INV=ROOT/'.generated/go2_memory_role_place_triplet_index_v1/train.jsonl'; OUT=Path('/home/andrewknowles/.cache/lewm_go2_temporal_v03/place_head_dev_v2'); SEED=2026081901
def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(1<<22),b''): h.update(b)
 return h.hexdigest()
class Head(nn.Module):
 def __init__(self):
  super().__init__(); self.m=nn.Sequential(nn.LayerNorm(1024),nn.Linear(1024,256),nn.GELU(),nn.Linear(256,128)); self.o=nn.Linear(256,128)
 def forward(self,x):
  z=self.m(x); return F.normalize(self.o(torch.cat([z.mean(1),z.amax(1)],1)),dim=-1)
class Trip(Dataset):
 def __init__(self,rows,c): self.rows=rows; self.c=c
 def __len__(self): return len(self.rows)
 def __getitem__(self,i):
  r=self.rows[i]; return tuple(self.c[p] for p in (r['anchor']['rgb_path'],r['positive']['rgb_path'],r['negative']['rgb_path']))
def main():
 random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED); OUT.mkdir(parents=True,exist_ok=True)
 rows=[json.loads(x) for x in open(INV)]; scenes=sorted({r['scene_id'] for r in rows},key=lambda s:hashlib.sha256(('split-v2'+s).encode()).hexdigest()); ev_s=set(s for i,s in enumerate(scenes) if i%4==0); fit=[r for r in rows if r['scene_id'] not in ev_s]; ev=[r for r in rows if r['scene_id'] in ev_s]
 paths=sorted({r[k]['rgb_path'] for r in rows for k in ('anchor','positive','negative')}); split={'seed':SEED,'fit_scenes':sorted({r['scene_id'] for r in fit}),'eval_scenes':sorted(ev_s),'fit_rows':len(fit),'eval_rows':len(ev),'unique_frames':len(paths),'missing':[p for p in paths if not Path(p).exists()]}; json.dump(split,open(OUT/'split.json','w'),sort_keys=True,indent=2)
 idx={}; cache={}; arm=VJepa21Arm(); device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'); arm.build(device,torch.float32); t0=time.time(); batch=16
 for j in range(0,len(paths),batch):
  todo=[]
  for p in paths[j:j+batch]:
   q=OUT/(hashlib.sha256(p.encode()).hexdigest()+'.npy')
   if q.exists():
    try:
     a=np.load(q,mmap_mode='r')
     if a.shape==(786432,):
      a=np.asarray(a).reshape(768,1024); np.save(q,a)
     if a.shape==(768,1024) and a.dtype==np.float16 and np.isfinite(a).all(): idx[p]=str(q); continue
    except Exception: pass
   todo.append((p,q))
  if todo:
   x=torch.stack([arm.preprocess(p) for p,_ in todo]).to(device)
   with torch.inference_mode():
    with torch.autocast(device_type='cuda',dtype=torch.bfloat16,enabled=device.type=='cuda'):
     y=arm.tokens(x).float().cpu().numpy().astype('float16')
   for (p,q),a in zip(todo,y): np.save(q,a); idx[p]=str(q)
  if j%160==0: json.dump(idx,open(OUT/'cache_index.json','w'),sort_keys=True); print('encoded/reused',j+len(paths[j:j+batch]),'/',len(paths),flush=True)
 json.dump(idx,open(OUT/'cache_index.json','w'),sort_keys=True); 
 for p,q in idx.items(): cache[p]=torch.from_numpy(np.load(q).copy()).float()
 model=Head().to(device); opt=torch.optim.AdamW(model.parameters(),lr=1e-3,weight_decay=1e-4); loader=DataLoader(Trip(fit,cache),batch_size=64,shuffle=True); losses=[]; t1=time.time()
 for ep in range(30):
  s=n=0
  for a,p,ng in loader:
   a,p,ng=a.to(device),p.to(device),ng.to(device); za,zp,zn=model(a),model(p),model(ng); loss=F.relu(.2+(za*zn).sum(1)-(za*zp).sum(1)).mean(); opt.zero_grad(); loss.backward(); opt.step(); s+=float(loss)*len(a); n+=len(a)
  losses.append(s/n); print('epoch',ep+1,losses[-1],flush=True)
 ck=OUT/'place_head_dev_v2.pt'; torch.save({'state_dict':model.state_dict(),'seed':SEED,'losses':losses},ck)
 result={'schema':'minimal_place_head_spike_v2','split':split,'cache_index':str(OUT/'cache_index.json'),'cache_index_sha256':sha(OUT/'cache_index.json'),'checkpoint':str(ck),'checkpoint_sha256':sha(ck),'losses':losses,'encoding_seconds':t1-t0,'training_seconds':time.time()-t1,'frames':len(idx)}; json.dump(result,open(OUT/'result.json','w'),sort_keys=True,indent=2); print(json.dumps(result,indent=2))
if __name__=='__main__': main()
