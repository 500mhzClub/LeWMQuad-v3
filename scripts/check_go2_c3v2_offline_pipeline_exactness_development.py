"""Pre-declared exactness gate for the offline C3 pipeline (C3-v2 readout fix, 29 Sep 2026).

1. Context construction: offline native contexts for C4 validation 11/0 equal the native
   contexts captured from the deployed controller by verified replay (c4_inputs).
2. Full path: offline contexts through the unchanged C3-v1 model forward reproduce C3-v1's
   logged run-time motion predictions on validation 11/0; the single-tape feature path used
   for readout training reproduces the forward for each candidate tape.
Reads preserved logs and replay-verified frames only; nothing is re-run on validation.
"""
import json
from pathlib import Path

import numpy as np
import torch

from lewm import decision_headroom_json_v42_development as output
from lewm.c3v2_offline_pipeline_development import Recording, pooled_features, readout_motion
from lewm_genesis.lewm_contract import apply_safety_limits_single
from scripts import run_go2_navigation_capability_completed_support_v4_development as owner

BASE = Path(json.loads(owner.PROTOCOL.read_text())['output_root'])
SAMPLE_STRIDE = 20


def validation_recording(arm):
    run = BASE/f'runs/v4_completed_support_validation_{arm}_val11_ep0_attempt001'
    video = BASE/f'videos/capability_{arm}_val11_ep0_attempt001/ego_frames'
    return run, Recording(run/'native/policy_histories.npz', run/'native/policy_observations.json',
                          lambda i: video/f'{i:04d}.png')


def main():
    output.install(BASE)
    report = {}
    # 1. Context construction against deployed C4 contexts.
    _, c4 = validation_recording('C4')
    captured = torch.load(BASE/'analysis/qualification_v4_2026-09-28/c4_inputs/v4_completed_support_validation_C4_val11_ep0_attempt001_inputs.pt',
                          weights_only=False)['inputs']
    worst = 0.
    for observed_ns, item in captured.items():
        frame = (observed_ns-1_500_000_000)//100_000_000
        mine, deployed = c4.native(frame), item['native']
        for key in ('pixels', 'context_times_ns', 'past_applied_commands', 'past_measured_ns', 'past_available_ns'):
            if not torch.equal(mine[key], deployed[key]):
                worst = max(worst, float((mine[key].double()-deployed[key].double()).abs().max()))
    report['context_construction'] = dict(decisions=len(captured), exact=worst == 0., maximum_abs=worst)
    # 2. Full C3-v1 path against logged run-time predictions.
    run, c3 = validation_recording('C3')
    receipts = json.loads((run/'model_calls.json').read_text())
    model = owner.source.load_dense_navigation_model('action', readout_arm='maze_view_maze_data')
    forward_worst = single_worst = 0.
    checked = 0
    for receipt in receipts[::SAMPLE_STRIDE]:
        frame = (receipt['observed_ns']-1_500_000_000)//100_000_000
        native = c3.native(frame)
        requested = torch.tensor(receipt['requested_commands'], dtype=torch.float32)
        blocks = (requested/torch.tensor([.3, 1., .5]))[:, :, None, :]
        model.pending_context = native
        model(observation_history=None, known_action_blocks=blocks, known_action_valid=torch.ones(6, 8, 1, dtype=torch.bool))
        recomputed = np.asarray(model.receipts[-1]['motion_xy_yaw'])
        logged = np.asarray(receipt['motion_xy_yaw'])
        forward_worst = max(forward_worst, float(np.abs(recomputed-logged).max()))
        last = tuple(native['past_applied_commands'][-1].tolist())
        requested_forward = (blocks[:, :, 0]*torch.tensor([.3, 1., .5])).numpy()  # exactly as the deployed forward
        for candidate in range(6):
            applied = np.asarray(apply_safety_limits_single(requested_forward[candidate].tolist(), last, model.limits)[0], np.float32)
            current, predicted = pooled_features(model, native, applied)
            single = np.stack([readout_motion(model.readout, current, predicted[h]) for h in range(1, 9)])
            single_worst = max(single_worst, float(np.abs(single-logged[candidate]).max()))
        checked += 1
    report['c3_v1_forward'] = dict(decisions=checked, maximum_abs_vs_logged=forward_worst, pass_1e6=forward_worst <= 1e-6)
    report['single_tape_path'] = dict(decisions=checked, candidate_tapes=6*checked, maximum_abs_vs_logged=single_worst, pass_1e6=single_worst <= 1e-6)
    report['passed'] = report['context_construction']['exact'] and report['c3_v1_forward']['pass_1e6'] and report['single_tape_path']['pass_1e6']
    out = BASE/'c3v2_rest_turn_recordings_v1'/'offline_pipeline_exactness.json'
    owner.save(out, report)
    print(json.dumps(report))


if __name__ == '__main__':
    main()
