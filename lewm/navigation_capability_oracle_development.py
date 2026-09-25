"""True motion in the existing prediction slot, with main-thread physics service.

No true map, current pose or branch contacts enter the controller. Branch poses
are converted to the same initial-body-relative motion targets as the readout.
"""
import math
import queue
import threading
import time

import numpy as np
import torch
from torch import nn

from lewm.decision_headroom_snapshot_development import capture, restore
from lewm.dense_native_observation_development import dense_native_context
from lewm.physical_execution_development import rotation_xyzw
from scripts.run_go2_decision_headroom_branches_development import RECORD_LISTS


class OracleMotionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.pending_context = None
        self.pending = queue.Queue()
        self.receipts = []
        self.physical_predictions = []
        self.readout_identity = dict(arm='C0_oracle', training_horizons_ms=list(range(100,801,100)),
                                    prediction_slot_only=True)
        self.eval()

    def set_native_context(self, packets, *, observed_ns):
        if self.pending_context is not None:
            raise ValueError('unconsumed oracle context')
        self.pending_context = dense_native_context(packets, observed_ns=observed_ns)

    def forward(self, *, observation_history, known_action_blocks, known_action_valid):
        native, self.pending_context = self.pending_context, None
        if native is None or known_action_blocks.shape != (6,8,1,3) or not known_action_valid.all():
            raise ValueError('unchanged complete dense candidate contract required')
        requested = (known_action_blocks[:,:,0].cpu()*torch.tensor([.3,1.,.5])).numpy()
        job = dict(requested=requested, observed_ns=int(native['context_times_ns'][-1]),
                   ready=threading.Event())
        self.pending.put(job)
        job['ready'].wait()
        if 'error' in job:
            raise job['error']
        return job['output']

    def service(self, session, budget):
        """Called by the collector on its native simulation thread while paused."""
        try:
            job = self.pending.get_nowait()
        except queue.Empty:
            return False
        try:
            job['output'] = self._branches(session, job, budget)
        except BaseException as exc:
            job['error'] = exc
            raise
        finally:
            job['ready'].set()
            self.pending.task_done()
        return True

    def _branches(self, session, job, budget):
        start = time.monotonic()
        snapshot = capture(session)
        if snapshot['measured_ns'] != job['observed_ns']:
            raise ValueError('oracle snapshot is not the source decision boundary')
        saved_lists = {name:getattr(session,name) for name in RECORD_LISTS if hasattr(session,name)}
        callback = session.physics_clock_callback
        initial = session.samples[-1]['base_pose_world'].copy()
        Q = rotation_xyzw(initial[3:])
        motions, branches, applied = [], [], []
        session.physics_clock_callback = None
        try:
            for candidate, requested in enumerate(job['requested']):
                restore(session, snapshot)
                for name in saved_lists:
                    setattr(session,name,[])
                tape = np.repeat(requested,5,axis=0)
                commands = []
                for command in tape:
                    budget.check()
                    commands.append(session.command_policy_step(command))
                poses = np.stack([row['base_pose_world'] for row in session.samples])
                assert poses.shape == (400,7)
                horizon = poses[np.arange(49,400,50)]
                xy = (horizon[:,:3]-initial[:3]) @ Q
                yaw = [math.atan2((Q.T@rotation_xyzw(p[3:]))[1,0],
                                  (Q.T@rotation_xyzw(p[3:]))[0,0]) for p in horizon]
                motions.append(np.c_[xy[:,:2],yaw])
                branches.append(dict(requested=tape.copy(), applied=np.asarray(commands), poses=poses))
                applied.append(np.asarray(commands)[::5].tolist())
        finally:
            restore(session, snapshot)
            for name,value in saved_lists.items():
                setattr(session,name,value)
            session.physics_clock_callback = callback
        motion = torch.tensor(np.asarray(motions),dtype=torch.float32)
        self.receipts.append(dict(observed_ns=job['observed_ns'], requested_commands=job['requested'].tolist(),
            applied_commands=applied, motion_xy_yaw=motion.tolist(), wall_s=time.monotonic()-start,
            native_step_ms=2, branch_simulated_s=4.8, contact_prediction_available=False))
        self.physical_predictions.append(dict(observed_ns=job['observed_ns'], branches=branches))
        result = torch.cat((motion[:,:,:2],motion[:,:,2:3].sin(),motion[:,:,2:3].cos(),
                            torch.full((6,8,1),-1000.)),dim=-1)
        return dict(rollout_outcomes=result, target_offsets_ns=torch.arange(1,9).mul(100_000_000).expand(6,8),
                    prediction_valid=torch.ones(6,8,dtype=torch.bool), contact_prediction_available=False)

    def verify_executed(self, requests, samples):
        """Match full native poses along every actually served candidate prefix.

        Overrides are explicit command-path departures; they are not evidence
        of oracle error. Every executed branch prefix must match continuously
        from its source snapshot, including the pre-dispatch commands.
        """
        stamps = np.rint(np.array([s['timestamp_s'] for s in samples])*1e9).astype(np.int64)
        poses = np.stack([s['base_pose_world'] for s in samples])
        request_by_time = {r['simulator_ns']:r for r in requests}
        rows = []
        for prediction in self.physical_predictions:
            origin = prediction['observed_ns']
            served = [r for r in requests if r.get('command_observation_ns') == origin]
            for branch_index, branch in enumerate(prediction['branches']):
                length = 0
                for tick in range(40):
                    row = request_by_time.get(origin+tick*20_000_000)
                    if row is None or not np.allclose(row['applied_command'],branch['applied'][tick],rtol=0,atol=1e-7):
                        break
                    length += 1
                indices = np.searchsorted(stamps, origin+np.arange(1,length*10+1)*2_000_000)
                if not length:
                    rows.append(dict(observed_ns=origin,candidate=branch_index,matched_prefix_ms=0,passed=False,
                                     reason='NO_MATCHING_EXECUTED_PREFIX'))
                    continue
                assert np.array_equal(stamps[indices],origin+np.arange(1,length*10+1)*2_000_000)
                actual, expected = poses[indices], branch['poses'][:length*10]
                position = np.linalg.norm(actual[:,:3]-expected[:,:3],axis=1)
                yaw = []
                for a,b in zip(actual,expected):
                    A,B=rotation_xyzw(a[3:]),rotation_xyzw(b[3:])
                    d=math.atan2(A[1,0],A[0,0])-math.atan2(B[1,0],B[0,0])
                    yaw.append(abs(math.atan2(math.sin(d),math.cos(d))))
                passed = float(position.max()) <= .001 and max(yaw) <= math.radians(.1)
                rows.append(dict(observed_ns=origin,candidate=branch_index,matched_prefix_ms=length*20,
                    maximum_position_error_m=float(position.max()),maximum_yaw_error_deg=math.degrees(max(yaw)),
                    passed=passed, served_command_samples=len(served)))
        return dict(schema='navigation_capability_oracle_prefix_check.v1',
            passed=bool(rows) and all(r['passed'] for r in rows), comparisons=rows,
            position_tolerance_m=.001,yaw_tolerance_degrees=.1,
            comparison='All command-matching native prefixes; overrides/changes terminate matching prefixes')
