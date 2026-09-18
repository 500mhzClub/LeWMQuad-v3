"""Counterfactual delayed plans on actual public histories from the independent maze."""
from collections import deque, Counter
from contextlib import closing
import json
from pathlib import Path
import time

import cv2
import numpy as np
import torch

from lewm.causal_subtrajectory_learning_development import causal_history_tensors
from lewm.delayed_action_planning_development import delayed_candidate_inputs, score_delayed_predictions
from lewm.observation_horizon_input_ablation_development import transform_inputs
from scripts import profile_stop_conditioned_early_decisions_development as source

BASE = source.trial.run.BASE
ROOT = BASE/'go2_stop_conditioned_independent_00_frozen_reference_seed_2026091001_full_jepa_v1_attempt_001'
INPUT = ROOT/'independent_00_frozen_reference_seed_2026091001_full_jepa'
OUTPUT = Path('docs/go2_delayed_action_model_probe_2026-09-13.json')
FAILURE = OUTPUT.with_suffix('.failure.json')


@torch.inference_mode()
def main():
    assert not OUTPUT.exists() and not FAILURE.exists()
    assert json.loads((ROOT/'result.json').read_text())['verified_round_trip'] is True
    cv2.setNumThreads(1);cv2.ocl.setUseOpenCL(False);torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    rows=[]; models={};states={}
    try:
        admission=json.loads((BASE/'go2_all_phase_adapter_maze02_matched_native_v1_attempt_001'/'launch.json').read_text())['input_admission']['correction_admission']
        for condition in ('direct','supervised_rollout','jepa'):
            model, actual, variant=source.load_assigned(admission,'seed_2026091001_full_'+condition)
            assert actual==condition and variant=='full'
            models[condition]=model;states[condition]=source.trial.previous.state_digest(model.state_dict())
        reader=source.packets.ExtendedReturnBudgetRGBDReplay(INPUT);history=deque(maxlen=4)
        with closing(source.packets.read_rows(INPUT)) as recorded:
            for frame in range(13):
                decision=next(recorded)['decision'];p,_,_,now=reader.packet(frame);history.append(p)
                if len(history)<4:continue
                causal=causal_history_tensors(list(history),now)
                goal=decision['new_selection']['goal_body_xy_m']
                inputs=transform_inputs(delayed_candidate_inputs(causal,[[0.,0.,0.],[0.,0.,0.]]),input_variant='full')
                for condition,model in models.items():
                    start=time.perf_counter();out=model(**inputs);seconds=time.perf_counter()-start
                    head='direct_outcomes' if condition=='direct' else 'rollout_outcomes'
                    assert out['prediction_valid'].all()
                    expected=torch.arange(1,9).mul(100_000_000).expand(6,8)
                    assert torch.equal(out['target_offsets_ns'],expected)
                    prediction=out[head].cpu().numpy()
                    selected=score_delayed_predictions(prediction,goal)
                    rows.append(dict(frame=frame,measured_ns=now,condition=condition,
                        inference_s=seconds,selection=selected,prediction=prediction.tolist()))
                print('DELAYED_ACTION_FRAME',frame,flush=True)
        for condition,model in models.items():
            assert source.trial.previous.state_digest(model.state_dict())==states[condition]
            assert all(p.grad is None for p in model.parameters())
        summary={condition:dict(observations=sum(r['condition']==condition for r in rows),
            median_inference_ms=float(np.median([r['inference_s'] for r in rows if r['condition']==condition]))*1000,
            selected_actions=dict(Counter(r['selection']['action'] for r in rows if r['condition']==condition)),
            maximum_common_prefix_spread=np.max([r['selection']['common_prefix_max_absolute_spread']
                for r in rows if r['condition']==condition],axis=0).tolist()) for condition in models}
        report=dict(status='DELAYED_ACTION_MODEL_PROBE_COMPLETE',rows=rows,summary=summary,
            model_state_sha256=states,models_unchanged=True,input=str(INPUT),
            input_result_sha256=source.digest(ROOT/'result.json'),
            committed_prefix=[[0.,0.,0.],[0.,0.,0.]],prefix_is_proposed_protocol_not_recorded_future=True,
            actual_historical_sensor_and_control_inputs=True,dispatch_delay_ns=200_000_000,
            candidate_duration_ns=100_000_000,known_zero_tail_ns=500_000_000,
            native_execution=False,prediction_accuracy_under_delayed_commands_measured=False,
            changed_trajectory_inferred=False,continuous_control_qualified=False,
            sources={p:source.digest(Path(p)) for p in ('lewm/delayed_action_planning_development.py',
                'scripts/probe_delayed_action_models_development.py')})
        with OUTPUT.open('x') as f:json.dump(report,f,indent=2);f.write('\n')
        print(json.dumps(summary),flush=True)
    except BaseException as error:
        with FAILURE.open('x') as f:json.dump(dict(reason=repr(error),completed_rows=rows),f,indent=2)
        raise


if __name__=='__main__':main()
