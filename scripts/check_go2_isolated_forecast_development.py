"""Check process-transfer forecasts against the same local frozen inference."""
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context
import hashlib
import json
from pathlib import Path
import time

import torch

from scripts.profile_go2_current_planning_development import ROOT, study, NoisyPublicReplay
from lewm.paced_multirate_controller_development import causal_history_tensors
from lewm.delayed_action_planning_development import delayed_candidate_inputs
from lewm.isolated_forecast_development import (
    initialize_forecast, forecast_ready, IsolatedForecastRuntime)
from lewm.pulse_timed_training_runner_development import state_digest
from lewm.shared_candidate_history_development import install_shared_history_encoding
from lewm.terminal_translation_pulse_development import command_sequences


class OfflineIsolatedRuntime(IsolatedForecastRuntime):
    def _worker(self, name, function):
        pass


@torch.inference_mode()
def main():
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    root = study.study.BASE / ROOT
    assert not (root / 'DEPTH_RETIRED').exists()
    output = study.study.BASE / 'go2_isolated_forecast_runtime_equivalence_v1_attempt_001'
    output.mkdir(exist_ok=False)
    reader = NoisyPublicReplay(root / 'native')
    plans = {p['frame']: p for p in json.loads((root / 'planning.json').read_text()) if 'selection' in p}
    rows, identities = [], []
    for arm in ('jepa', 'supervised_rollout'):
        local, _, _ = study.study.load_model(arm)
        install_shared_history_encoding(local)
        with ProcessPoolExecutor(max_workers=1, mp_context=get_context('spawn'),
                initializer=initialize_forecast, initargs=(arm,)) as executor:
            identity = executor.submit(forecast_ready).result(timeout=60.)
            assert identity['state_sha256'] == state_digest(local.state_dict())
            identities.append(identity)
            parent_model, condition, variant = study.study.load_model(arm)
            runtime = OfflineIsolatedRuntime(parent_model, forecast_executor=executor,
                forecast_identity=identity, prediction_source='neural', condition=condition,
                variant=variant, registration_executor=None, mapping_executor=None,
                pose_executor=None, obstacle_executor=None, goal_initial_xy=[0., 1.3],
                clock_ns=lambda: 0, planning_delay_ticks=3, navigation_ticks=4800,
                arrival_radius_m=.02)
            for frame in (300, 700, 1100):
                plan = plans[frame]
                history = [reader.packet(f)[0] for f in range(frame-3, frame+1)]
                inputs = delayed_candidate_inputs(causal_history_tensors(history, plan['measured_ns']),
                    plan['committed_prefix'], delay_ticks=3, commit_ticks=4)
                for pulse in (False, True):
                    inputs['known_action_blocks'] = torch.as_tensor(command_sequences(
                        plan['committed_prefix'], pulse=pulse)[:, :, None], dtype=torch.float32) / torch.tensor([.3, 1., .5])
                    expected = local(**inputs)
                    begin = time.perf_counter_ns()
                    actual = runtime.model(**inputs)
                    transfer_ms = (time.perf_counter_ns()-begin)/1e6
                    for key in expected:
                        torch.testing.assert_close(expected[key], actual[key], atol=0, rtol=0)
                    rows.append(dict(arm=arm, frame=frame, terminal_pulse=pulse,
                        all_outputs_bitwise_equal=True, elapsed_including_transfer_ms=transfer_ms,
                        worker_receipt=runtime.forecast_receipts[-1]))
            distinct = {k: v.clone() for k, v in inputs['observation_history'].items()}
            distinct['rgb'][1].zero_()
            inputs = dict(inputs, observation_history=distinct)
            expected = local(**inputs)
            actual = runtime.model(**inputs)
            for key in expected:
                torch.testing.assert_close(expected[key], actual[key], atol=0, rtol=0)
            assert runtime.shared_history_receipt == dict(shared_calls=6, ordinary_calls=1)
            assert runtime.model_input_calls == 7 and len(runtime.forecast_receipts) == 7
    sources = {}
    for name in (__file__, 'lewm/isolated_forecast_development.py'):
        data = Path(name).read_bytes(); sources[name] = hashlib.sha256(data).hexdigest()
        (output / Path(name).name).write_bytes(data)
    result = dict(status='PASS', rows=rows, identities=identities,
        distinct_history_fallback_bitwise_equal=True, source_sha256=sources,
        runtime_model_hooks_and_receipts_tested=True,
        live_concurrency_tested=False, navigation_benefit_demonstrated=False)
    (output / 'result.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    main()
