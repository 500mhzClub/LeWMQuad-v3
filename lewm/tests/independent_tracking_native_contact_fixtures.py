"""Synthetic solver state for guard tests, never native evidence."""
from types import SimpleNamespace as NS
import numpy as np

from scripts.independent_tracking_native_contact_guard_development import NativeContactGuard


def runtime():
    state=NS(error=None,calls=0)
    def check():
        state.calls+=1
        if state.error:raise RuntimeError(state.error)
    info=NS(max_possible_pairs={None:200},max_collision_pairs={None:150},max_contact_pairs={None:750})
    solver=NS(n_envs=1,max_collision_pairs=150,
        _static_rigid_sim_config=NS(requires_grad=False),
        _options=NS(max_collision_pairs=150,box_box_detection=False,enable_collision=True,enable_self_collision=True),
        collider=NS(_collider_info=info,_collider_static_config=NS(n_contacts_per_pair=5)),check_errno=check)
    gs=NS(backend='cpu_fixture',cpu='cpu_fixture',np_float=np.float32,np_int=np.int32)
    return solver,gs,state


def report_for_count(n):
    # Synthetic prebuilt cohort fixtures do not execute n native steps. Only
    # the dedicated guard/recorder tests exercise actual per-sample callbacks.
    solver,gs,_=runtime();guard=NativeContactGuard(solver,gs)
    guard.passed=guard.attempted=n;guard.finish(n)
    return guard.report()


class MockCollectionGuard:
    """Only collection lifecycle tests; no simulated/native physics is run."""
    def __init__(self):self.row=None;self.error=None
    def finish(self,n):
        if self.error:raise RuntimeError(self.error)
        self.row=report_for_count(n)
    def report(self):return self.row
