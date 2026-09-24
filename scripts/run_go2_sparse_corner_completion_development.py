"""One exposed-layout JEPA navigation test of sparse corner completion."""
import argparse
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

from lewm.eligible_floor_registration_development import bind
from lewm.axis_aligned_fine_connectivity_development import warmup
from lewm.projected_polygon_floor_coverage_development import initialize_mapping
from lewm.sparse_corner_completion_runtime_development import (
    SparseCornerCompletionMotion, SparseCornerCompletionRuntimeMixin)
from scripts import run_go2_route_turn_memory_transfer_development as transfer

previous = transfer.previous
BASE = transfer.BASE
ROOT = 'go2_sparse_corner_completion_jepa_noise_2mm_native_layout01_4800_v1_attempt_001'
PLAN = Path('docs/go2_sparse_corner_completion_plan_2026-09-17.json')
RAW_WRITE = transfer.RAW_WRITE
OUTPUT = None
def initialize_pose(output):
    bind(transfer.native.baseline.initialize_pose,
        CadencedViewRevisitMotion=SparseCornerCompletionMotion)(output)


native = SimpleNamespace(**(vars(transfer.native) | dict(baseline=SimpleNamespace(
    **(vars(transfer.native.baseline) | dict(initialize_pose=initialize_pose))))))


class CompletionRuntime(SparseCornerCompletionRuntimeMixin, transfer.TransferRuntime):
    pass


def root_name(number):
    assert number == 1
    return ROOT


def sources():
    return transfer.sources() | {p: hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in (
        'lewm/sparse_feature_budget_tracking_development.py',
        'lewm/sparse_corner_completion_development.py',
        'lewm/sparse_corner_completion_runtime_development.py',
        'lewm/tests/test_sparse_corner_completion_runtime_development.py',
        'scripts/run_go2_sparse_corner_completion_development.py')}


def write(name, value):
    if name == 'launch.json':
        value = value | dict(experiment='sparse_corner_completion_v1',
            actual_mapping_class='ProjectedPolygonFloorRoutingMap',
            projected_polygon_floor_coverage=True, route_turn_memory_enabled=True,
            axis_routing_acceleration_enabled=True, planned_native_assignments=1,
            sparse_corner_completion_enabled=True, total_tracking_feature_cap=150,
            visual_recovery_uses_original_strong_corner_counts=True,
            tracker_pose_acceptance_thresholds_unchanged=True,
            reference_root_path=str(BASE/transfer.root_name(1)),
            reference_root_name=transfer.root_name(1),
            world_model_changed=False, models_are_same_frozen_readouts=True,
            controller_unchanged_from_repeatability_batch=False,
            exposed_development_layout=True, new_independent_development_layout=False,
            final_evaluation=False, hardware_validated=False)
    bind(RAW_WRITE, OUTPUT=OUTPUT)(name, value)


def prepare():
    assert not PLAN.exists() and not (BASE/ROOT).exists()
    frozen = json.loads(transfer.PLAN.read_text())
    assert transfer.sources() == frozen['source_sha256']
    replay = BASE/transfer.root_name(1)/'sparse_corner_completion_replay_v1'
    result = json.loads((replay/'result.json').read_text())
    accuracy = json.loads((replay/'pose_accuracy_v1.json').read_text())
    assert result['complete_saved_sensor_sequence_accepted'] and result['accepted_frames'] == 1272
    for path, digest in result['source_sha256'].items():
        assert hashlib.sha256(Path(path).read_bytes()).hexdigest() == digest
    plan = frozen | dict(schema='sparse_corner_completion_plan.v1',
        assignments=[[1, 'jepa']], root_names=[ROOT], planned_native_assignments=1,
        models={'jepa': frozen['models']['jepa']}, source_sha256=sources(),
        actual_runtime_class=CompletionRuntime.__name__,
        reference_root=str(BASE/transfer.root_name(1)),
        intervention='fill spare sparse-view feature slots with measured weaker corners; retain old strong-corner recovery signal',
        replay_accepted_frames=1272, replay_extension_frames=5,
        replay_common_xy_rmse_mm=accuracy['changed_on_common']['xy_rmse_mm'],
        exposed_development_layout=True, new_independent_development_layout=False,
        fixed_before_first_execution=True, no_extra_repetitions_to_obtain_success=True,
        unchanged=['frozen JEPA readout and six candidates', '150-feature matching cap',
            'all pose acceptance thresholds', 'strong-feature visual recovery signal and thresholds',
            'route-turn memory, polygon geometry, axis routing acceleration',
            '2mm depth noise, ideal gyro, 4800 ticks, CPU group, deadlines'],
        limitations=['one now-exposed maze', 'replay extends only five frames beyond old failure',
            'trajectory and timing may differ', 'no isolated reliability or JEPA advantage claim',
            'measured simulation; no hardware validation'])
    previous.previous.save(PLAN, plan)
    print('PREPARED one sparse-corner completion native pilot', flush=True)


def run():
    warmup()
    original_mapper = previous.study.cohort.learned.initialize_mapping
    original_writer = previous.study.source.write
    previous.study.cohort.learned.initialize_mapping = initialize_mapping
    previous.study.source.write = write
    try:
        bind(previous.run, PLAN=PLAN, ARMS=('jepa',), root_name=root_name,
            source_hashes=sources, Runtime=CompletionRuntime, native=native)(1)
    finally:
        previous.study.cohort.learned.initialize_mapping = original_mapper
        previous.study.source.write = original_writer


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--prepare', action='store_true')
    group.add_argument('--evaluate', action='store_true')
    args = parser.parse_args()
    if args.prepare:
        prepare()
    elif args.evaluate:
        bind(previous.evaluate, PLAN=PLAN, ARMS=('jepa',), root_name=root_name)(1)
    else:
        run()
