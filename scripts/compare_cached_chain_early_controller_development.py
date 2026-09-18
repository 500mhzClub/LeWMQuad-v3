"""Paired complete decisions and phase timing for the combined tracker candidate."""
from contextlib import closing
import json
from pathlib import Path
import time

import cv2
import numpy as np
import psutil
import torch

from scripts import profile_stop_conditioned_early_decisions_development as source
from scripts.compare_full_consensus_recorded_tracker_development import without_work_counts
from lewm.cached_chain_tracker_development import CachedChainVisualMotion
from lewm.sampled_plane_stop_conditioned_controller_development import SampledPlaneStopConditionedController
from lewm.controller_phase_timing_development import PhaseTiming, model_forward_timing

OUTPUT = Path('docs/go2_cached_chain_early_controller_2026-09-13.json')
FAILURE = OUTPUT.with_suffix('.failure.json')


class Candidate(SampledPlaneStopConditionedController):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.motion = CachedChainVisualMotion()


def instrument(controller, timing):
    for component, name in ((controller.motion, 'tracking'),
            (controller.registration, 'floor_registration'), (controller.mapper, 'mapping'),
            (controller.selector, 'selection')):
        method = 'choose' if name == 'selection' else 'observe'
        bound = getattr(component, method)
        def measured(*args, _bound=bound, _name=name, **kwargs):
            with timing.scope(_name): return _bound(*args, **kwargs)
        setattr(component, method, measured)


def main():
    assert not OUTPUT.exists() and not FAILURE.exists()
    assert psutil.virtual_memory().available > 16*1024**3
    cv2.setNumThreads(1); cv2.ocl.setUseOpenCL(False); torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    rows = []
    try:
        admission_root = source.trial.run.BASE/'go2_all_phase_adapter_maze02_matched_native_v1_attempt_001'
        admission = json.loads((admission_root/'launch.json').read_text())['input_admission']['correction_admission']
        model, condition, variant = source.load_assigned(admission, 'seed_2026091001_no_rgb_direct')
        before = source.trial.previous.state_digest(model.state_dict())
        assert before == json.loads((source.ROOT/'launch.json').read_text())['model_state_sha256']
        mission = json.loads((source.INPUT/'public_mission.json').read_text())
        arms = [cls(model, source.trial.previous.geometry_factory(source.trial.previous.URDF),
            public_mission=mission, navigation_ticks=8000, condition=condition,
            variant=variant, persistent=True) for cls in (source.StopConditionedSettlingController, Candidate)]
        timing = PhaseTiming()
        for arm in arms: instrument(arm, timing)
        reader = source.packets.ExtendedReturnBudgetRGBDReplay(source.INPUT)
        auxiliary = json.loads((source.INPUT/'auxiliary_camera_audit.json').read_text())
        with model_forward_timing(model, timing), closing(source.packets.read_rows(source.INPUT)) as recorded:
            for frame in range(13):
                expected = without_work_counts(next(recorded)['decision'])
                p, d, f, now = reader.packet(frame)
                image, depth = source.packets.rgb_packet(source.INPUT, frame, p,
                    source.public_acquisition(auxiliary[frame]), now_ns=now)
                result = dict(frame=frame, timed=frame>=3)
                for index in ((0, 1) if frame%2 == 0 else (1, 0)):
                    label = ('baseline', 'candidate')[index]; timing.reset()
                    started = time.perf_counter()
                    actual = arms[index].observe(p, d, f, now_ns=now,
                        auxiliary_depth=depth, auxiliary_rgb=image)
                    result[label+'_s'] = time.perf_counter()-started
                    result[label+'_phases'] = timing.snapshot()
                    if index == 1: assert actual.pop('sampled_plane_candidates_enabled') is True
                    if without_work_counts(json.loads(json.dumps(actual))) != expected:
                        raise ValueError(f'controller decision differs beyond work counts: frame={frame}, arm={label}')
                result['complete_decisions_equal_except_work_counts'] = True
                rows.append(result)
                print('WHOLE_CONTROLLER_FRAME', frame, result['baseline_s'], result['candidate_s'], flush=True)
        assert source.trial.previous.state_digest(model.state_dict()) == before
        assert all(p.grad is None for p in model.parameters())
        timed = rows[3:]; totals = {label:sum(r[label+'_s'] for r in timed) for label in ('baseline','candidate')}
        phases = {label:{phase:sum(r[label+'_phases'].get(phase, {}).get('exclusive_ns', 0)
            for r in timed)/len(timed)/1e6 for phase in ('tracking','floor_registration','mapping','selection','model.forward')}
            for label in ('baseline','candidate')}
        report = dict(status='CACHED_CHAIN_EARLY_CONTROLLER_COMPLETE', rows=rows, total_s=totals,
            reduction_percent=100*(1-totals['candidate']/totals['baseline']),
            median_ms={label:float(np.median([r[label+'_s'] for r in timed]))*1000 for label in totals},
            mean_exclusive_phase_ms=phases, model_state_sha256=before, model_unchanged=True,
            input=str(source.INPUT), shared_host=True, full_journey_tracker_running_concurrently=True,
            ignored_comparison_key='valid_proposals', early_history_only=True,
            full_history_controller_equivalence_proven=False, native_execution=False,
            acquisition_included_in_timing=False, continuous_execution_qualified=False, adopted=False,
            sources={p:source.digest(Path(p)) for p in ('lewm/cached_chain_association_development.py',
                'lewm/cached_chain_tracker_development.py', 'scripts/compare_cached_chain_early_controller_development.py')})
        with OUTPUT.open('x') as out: json.dump(report,out,indent=2);out.write('\n')
        print(json.dumps({k:v for k,v in report.items() if k!='rows'}),flush=True)
    except BaseException as error:
        with FAILURE.open('x') as out: json.dump(dict(reason=repr(error),completed_rows=rows),out,indent=2)
        raise


if __name__ == '__main__': main()
