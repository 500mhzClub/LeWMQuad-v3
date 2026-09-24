"""Synchronous frozen-model inference in one spawned CPU process.

Only causal observation tensors and proposed commands cross this boundary.
The existing measured planning clock includes packing, transfer, inference,
and response waiting; command deadlines are unchanged.
"""
import os
import time

import torch

from lewm.live_planning_stage_profile_development import LivePlanningStageProfileRuntime
from lewm.pulse_timed_training_runner_development import state_digest
from lewm.shared_candidate_history_development import install_shared_history_encoding


_model = None
_shared = None
_identity = None
_calls = 0


def initialize_forecast(arm):
    global _model, _shared, _identity, _calls
    from scripts.run_go2_persistent_visual_learning_comparison_development import study
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    _model, condition, variant = study.load_model(arm)
    _shared = install_shared_history_encoding(_model)
    _calls = 0
    _identity = dict(pid=os.getpid(), arm=arm, condition=condition, variant=variant,
        state_sha256=state_digest(_model.state_dict()), torch_threads=torch.get_num_threads(),
        cpu_affinity=sorted(os.sched_getaffinity(0)), shared_history_encoding=True)


def forecast_ready():
    if _model is None or _model.training:
        raise RuntimeError('frozen inference worker is not ready')
    return _identity


def forecast_payload(observation_history, known_action_blocks, known_action_valid):
    shared = all(v.shape[0] == 6 and v.stride(0) == 0 for v in observation_history.values())
    return dict(history={k: (v[:1] if shared else v).detach().numpy()
                         for k, v in observation_history.items()},
        broadcast_six=shared, blocks=known_action_blocks.detach().numpy(),
        valid=known_action_valid.detach().numpy())


@torch.inference_mode()
def forecast_numpy(payload):
    global _calls
    wall, cpu = time.perf_counter_ns(), time.thread_time_ns()
    history = {k: torch.from_numpy(v) for k, v in payload['history'].items()}
    if payload['broadcast_six']:
        history = {k: v.expand(6, *v.shape[1:]) for k, v in history.items()}
    result = _model(observation_history=history,
        known_action_blocks=torch.from_numpy(payload['blocks']),
        known_action_valid=torch.from_numpy(payload['valid']))
    outputs = {k: v.numpy() for k, v in result.items()}
    _calls += 1
    return dict(outputs=outputs, receipt=dict(call=_calls, pid=os.getpid(),
        worker_cpu_ns=time.thread_time_ns()-cpu,
        worker_wall_ns=time.perf_counter_ns()-wall, **_shared))


class IsolatedForecastRuntime(LivePlanningStageProfileRuntime):
    def __init__(self, *args, forecast_executor, forecast_identity, **kwargs):
        super().__init__(*args, **kwargs)
        if state_digest(self.model.state_dict()) != forecast_identity['state_sha256']:
            raise ValueError('worker must use the identical frozen model')
        self.forecast_executor = forecast_executor
        self.forecast_identity = forecast_identity
        self.forecast_receipts = []
        self.model.forward = self._remote_forward

    @property
    def shared_history_receipt(self):
        if not self.forecast_receipts:
            return dict(shared_calls=0, ordinary_calls=0)
        return {k: self.forecast_receipts[-1][k] for k in ('shared_calls', 'ordinary_calls')}

    def _remote_forward(self, observation_history, known_action_blocks, known_action_valid):
        return self._measure('isolated_forward_including_transfer', self._forecast,
            observation_history, known_action_blocks, known_action_valid)

    def _forecast(self, observation_history, known_action_blocks, known_action_valid):
        payload = forecast_payload(observation_history, known_action_blocks, known_action_valid)
        response = self.forecast_executor.submit(forecast_numpy, payload).result(timeout=5.)
        receipt = response['receipt']
        if receipt['pid'] != self.forecast_identity['pid'] or receipt['call'] != len(self.forecast_receipts)+1:
            raise RuntimeError('unexpected inference worker or response order')
        self.forecast_receipts.append(receipt)
        return {k: torch.from_numpy(v) for k, v in response['outputs'].items()}
