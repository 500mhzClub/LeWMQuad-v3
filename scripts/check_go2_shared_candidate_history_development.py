"""Compare frozen-model forecasts on recorded causal inputs before native use."""
import copy
import hashlib
import json
from pathlib import Path
import statistics
import time

import torch

from scripts.profile_go2_current_planning_development import ROOT, study, NoisyPublicReplay
from lewm.paced_multirate_controller_development import causal_history_tensors
from lewm.delayed_action_planning_development import delayed_candidate_inputs
from lewm.shared_candidate_history_development import install_shared_history_encoding
from lewm.terminal_translation_pulse_development import command_sequences


@torch.inference_mode()
def main():
    torch.set_num_threads(1)
    root = study.study.BASE / ROOT
    assert not (root / 'DEPTH_RETIRED').exists()
    output = study.study.BASE / 'go2_shared_candidate_history_equivalence_v1_attempt_001'
    output.mkdir(exist_ok=False)
    reader = NoisyPublicReplay(root / 'native')
    plans = {p['frame']: p for p in json.loads((root / 'planning.json').read_text()) if 'selection' in p}
    rows = []
    for arm in ('jepa', 'supervised_rollout'):
        original, _, _ = study.study.load_model(arm)
        optimized = copy.deepcopy(original)
        receipt = install_shared_history_encoding(optimized)
        for frame in (300, 700, 1100):
            plan = plans[frame]
            history = [reader.packet(f)[0] for f in range(frame - 3, frame + 1)]
            inputs = delayed_candidate_inputs(causal_history_tensors(history, plan['measured_ns']),
                plan['committed_prefix'], delay_ticks=3, commit_ticks=4)
            pulse = plan['motion_correction']['terminal_translation_pulse']
            if pulse:
                inputs['known_action_blocks'] = torch.as_tensor(command_sequences(
                    plan['committed_prefix'], pulse=True)[:, :, None], dtype=torch.float32) / torch.tensor([.3, 1., .5])
            a, b = original(**inputs), optimized(**inputs)
            errors = {}
            for key in a:
                torch.testing.assert_close(a[key], b[key], atol=1e-6, rtol=1e-6)
                if a[key].is_floating_point():
                    errors[key] = float((a[key] - b[key]).abs().max())
            timings = {}
            for label, model in (('original', original), ('shared_history', optimized)):
                values = []
                for _ in range(10):
                    begin = time.perf_counter_ns(); model(**inputs)
                    values.append((time.perf_counter_ns() - begin) / 1e6)
                timings[label] = statistics.median(values)
            rows.append(dict(arm=arm, frame=frame, terminal_pulse=pulse,
                maximum_absolute_errors=errors, median_forward_ms=timings))
        # Non-broadcast observations must never be silently replaced by row 0.
        distinct = {k: v.clone() for k, v in inputs['observation_history'].items()}
        distinct['rgb'][1].zero_()
        inputs = dict(inputs, observation_history=distinct)
        a, b = original(**inputs), optimized(**inputs)
        for key in a:
            torch.testing.assert_close(a[key], b[key], atol=0, rtol=0)
        assert receipt == dict(shared_calls=33, ordinary_calls=1), receipt
    sources = {}
    for name in (__file__, 'lewm/shared_candidate_history_development.py'):
        data = Path(name).read_bytes(); sources[name] = hashlib.sha256(data).hexdigest()
        (output / Path(name).name).write_bytes(data)
    result = dict(status='PASS', rows=rows, distinct_history_fallback_exact=True,
        atol=1e-6, rtol=1e-6, same_frozen_weights=True, source_sha256=sources,
        native_timing_benefit_demonstrated=False, navigation_benefit_demonstrated=False)
    (output / 'result.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    main()
