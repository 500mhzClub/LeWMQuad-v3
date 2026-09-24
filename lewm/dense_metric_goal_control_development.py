"""Same dense visual controller with a frozen learned goal distance."""
import time

import numpy as np
import torch
import torch.nn.functional as F

from lewm.dense_visual_goal_control_development import DenseVisualGoalControl
from lewm.dense_visual_motion_readout_development import pool_tokens
from lewm.geometry_progress_pilot_development import ACTIONS, candidate_commands
from lewm_genesis.lewm_contract import apply_safety_limits_single
from scripts import train_go2_dense_goal_metric_development as fitted


class MetricGoalControl(DenseVisualGoalControl):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.metric=fitted.load().cuda()
        with torch.inference_mode():
            self.goal_embedding=self.metric.embed(pool_tokens(self.goal[None]))

    @torch.inference_mode()
    def observe(self,packet):
        record=super().observe(packet)
        embedding=self.metric.embed(pool_tokens(self.history[-1][1][None]))
        record['current_goal_metric']=float((embedding-self.goal_embedding).square().mean())
        return record

    @torch.inference_mode()
    def choose(self,packet):
        start=time.monotonic();assert len(self.history)==3
        now=int(packet['image']['measured_ns']);assert self.history[-1][0]==now
        control=packet['sensor_state']['control']['applied_command']
        assert np.asarray(control['valid']).all()
        assert np.asarray(control['measured_ns'])[[4,9,14]].tolist()==[t for t,_ in self.history]
        assert (np.asarray(control['available_ns'])<=now).all()
        commands=np.asarray(control['values'],np.float32);assert np.all(commands[:,1]==0)
        requested=np.asarray([[candidate_commands(a)[0]]*5 for a in ACTIONS],np.float64)
        applied=np.asarray([apply_safety_limits_single(v.tolist(),tuple(commands[-1]),self.limits)[0]
                            for v in requested],np.float32)
        x=torch.stack([f for _,f in self.history])[None]
        c=torch.from_numpy((commands[:,[0,2]].reshape(3,5,2)-self.mean)/self.std).cuda()[None]
        a=torch.from_numpy(applied[:,:,[0,2]].reshape(6,10)).cuda()
        mask=torch.ones(6,768,dtype=torch.bool,device='cuda')
        if self.arm=='no_future_action':
            p=self.model(x,torch.zeros_like(a[:1]),mask[:1],control=c)
        else:
            p=self.model(x.expand(6,-1,-1,-1),a,mask,control=c.expand(6,-1,-1,-1))
        prediction=F.layer_norm(p.float(),(1024,));assert torch.isfinite(prediction).all()
        embedding=self.metric.embed(pool_tokens(prediction))
        costs=(embedding-self.goal_embedding).square().mean(-1)
        raw=(prediction-self.goal).square().mean((-1,-2))
        if self.arm=='no_future_action':
            costs=costs.expand(6);raw=raw.expand(6);prediction=prediction.expand(6,-1,-1)
        costs=costs.cpu().numpy();assert np.isfinite(costs).all()
        ties=np.flatnonzero(costs==costs.min());index=int(self.rng.choice(ties))
        self.pending=(now,prediction[index].clone(),self.history[-1][1],applied[index])
        return dict(observed_ns=now,action=ACTIONS[index],action_index=index,
            costs=costs.tolist(),cost_kind='learned_physical_goal_metric',raw_goal_mse_costs=raw.cpu().tolist(),
            tied_indices=ties.tolist(),requested_commands=requested[index].tolist(),
            expected_applied_commands=applied[index].tolist(),forecast_horizon_ms=500,commit_ticks=5,
            planning_wall_s=time.monotonic()-start)
