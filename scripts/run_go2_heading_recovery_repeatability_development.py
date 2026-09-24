"""Four predeclared repetitions of the unchanged heading-recovery controller."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
from multiprocessing import get_context
import os
from pathlib import Path
from types import SimpleNamespace

from lewm.eligible_floor_registration_development import bind
from lewm.process_registered_round_trip_development import registration_ready
from scripts import run_go2_hold_relative_heading_recovery_development as original


def repeat_write(name, value):
    if name == 'launch.json':
        reference_name = f'go2_hold_relative_heading_recovery_native_layout{LAYOUT_INDEX:02d}_4800_v1_attempt_001'
        reference = json.loads((BASE/reference_name/'launch.json').read_text())
        changed = [k for k in reference.keys()|value.keys()
            if k not in ('owner', 'extra_sources') and reference.get(k)!=value.get(k)]
        if changed or value['extra_sources'] != reference['extra_sources']:
            raise ValueError(f'frozen repeatability controller/settings differ: {changed}')
        value = value | dict(repeatability_series='heading_recovery_fixed_four_repeats_v1',
            repetition_index=REPETITION_INDEX, reference_root_name=reference_name,
            new_independent_layout=False,
            extra_sources=value['extra_sources'] | {
                __file__:hashlib.sha256(Path(__file__).read_bytes()).hexdigest()})
    RAW_WRITE(name, value)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--layout-index', type=int, choices=(0,1), required=True)
    parser.add_argument('--repetition-index', type=int, choices=(1,2), required=True)
    args = parser.parse_args(); prior = original.prior
    if sorted(os.sched_getaffinity(0)) != prior.CPU_GROUPS[args.layout_index]:
        raise ValueError('assigned physical CPU group required')
    stable = prior.prior.fresh.stable; layouts = prior.prior.fresh.layouts
    assignment = 'seed_2026091001_full_jepa'
    output = stable.source.BASE/f'go2_heading_recovery_repeatability_rep{args.repetition_index}_layout{args.layout_index:02d}_4800_v1_attempt_001'
    stable.source.validate_root(output, must_exist=False)
    if output.exists():
        raise ValueError('preserve completed or partial repetition')
    root, sha = prior.prior.FITS['jepa']
    repeated = bind(repeat_write, LAYOUT_INDEX=args.layout_index,
        REPETITION_INDEX=args.repetition_index, BASE=stable.source.BASE,
        RAW_WRITE=bind(stable.source.write, OUTPUT=output))
    annotated = bind(original.annotate_write, LAYOUT_INDEX=args.layout_index, RAW_WRITE=repeated)
    final_write = bind(prior.finish_write, LAYOUT_INDEX=args.layout_index, CONDITION='directed',
        CORRECTION_ROOT=root, CORRECTION_HASH=sha, RAW_WRITE=annotated)
    stable_writer = bind(stable.write, OUTPUT=output, REACTIVE=False,
        ARRIVAL_ENTRY_PRIORITY=True, source=SimpleNamespace(write=final_write))
    writer = bind(stable.configuration.write, _write=stable_writer,
        MODEL_ASSIGNMENT=assignment, USE_CLEARANCE_TURN_RECOVERY=True,
        USE_STEPWISE_TURN_RECOVERY=True, USE_TERMINAL_POSITION_PRIORITY=True,
        USE_PROGRESS_REJOINING=True, USE_PREDICTIVE_ARRIVAL_HOLD=True)
    stable.floor.configure()
    with ProcessPoolExecutor(max_workers=1, mp_context=get_context('spawn'),
            initializer=stable.floor.initialize_registration) as executor:
        assert executor.submit(registration_ready).result()

        def runtime(*runtime_args, **kwargs):
            if kwargs.get('condition') != 'jepa':
                raise ValueError('fixed JEPA model required')
            return original.HeadingRecoveryRuntime(*runtime_args,
                registration_executor=executor, navigation_ticks=4800, arrival_radius_m=.02, **kwargs)

        bind(stable.source.main, OUTPUT=output, COUNT=4814, LAYOUT_INDEX=args.layout_index,
            specification=layouts.specification, public_mission=layouts.public_mission,
            MODEL_ASSIGNMENT=assignment, PacedNativeSession=prior.CompactFreshCameraSession,
            write=writer, CLOCK_MODE='measured_simulation', PLANNING_DELAY_TICKS=3,
            POSE_INITIALIZER=stable.initialize_pose, MEASURED_RUNTIME_CLASS=runtime,
            OBSTACLE_INITIALIZER=stable.floor.initialize_obstacles,
            OBSTACLE_READY=stable.obstacles_ready,
            initialize_mapping=stable.floor.initialize_mapping)()


if __name__ == '__main__': main()
