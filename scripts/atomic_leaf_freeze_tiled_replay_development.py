"""Original full-history replay body with only atomic-leaf-freeze controller bindings.

This module has no launch entry point. Its caller must authenticate the ended
tiled predecessor, bind original raw/model inputs, and create the sole
exclusive output before invoking replay.
"""
import builtins
from types import FunctionType, SimpleNamespace
from lewm.tiled_density_progressive_floor_controller_development import TiledDensityProgressiveFloorController
from lewm.atomic_leaf_freeze_tiled_controller_development import AtomicLeafFreezeTiledController, normalize_to_tiled
from scripts import replay_go2_tiled_density_progressive_floor_late_history_v1 as previous
from scripts.progressive_batched_floor_state_development import normalized_state_tree, STATE_TYPE_PATHS
from scripts.navigation_artifact_root_development import BASE

SOURCE='scripts/atomic_leaf_freeze_tiled_replay_development.py'
TEST='lewm/tests/test_atomic_leaf_freeze_tiled_replay_development.py'
OUTPUT=BASE/'go2_atomic_leaf_freeze_tiled_late_history_v1_attempt_001'
original=previous.original


def normalize_candidate(decision):
    return previous.harness.normalize_candidate(normalize_to_tiled(decision))


def progress(*args,**kwargs):
    if args and args[0]=='SCOPED_FOOTPRINT_PAIRED_FRAME':
        args=('ATOMIC_LEAF_FREEZE_TILED_PAIRED_FRAME',*args[1:])
    builtins.print(*args,**kwargs)


def isolated_replay():
    function=original.replay
    view=SimpleNamespace(**vars(original.profile))
    view.normalize_candidate=previous.harness.normalize_candidate
    namespace=dict(function.__globals__,profile=view,
        FrozenFootprintAnchoredController=TiledDensityProgressiveFloorController,
        ScopedFootprintAnchoredController=AtomicLeafFreezeTiledController,
        normalize_candidate=normalize_candidate,state_tree=normalized_state_tree,
        OUTPUT=OUTPUT,print=progress)
    clone=FunctionType(function.__code__,namespace,function.__name__,function.__defaults__,function.__closure__)
    clone.__kwdefaults__=function.__kwdefaults__
    return clone


def replay(rows,prior_report):
    report=isolated_replay()(rows)
    if report['observed_state_checks']!=prior_report['observed_state_checks']:
        raise ValueError('all seven completed tiled retained-state identities required')
    return report|dict(baseline='TiledDensityProgressiveFloorController',candidate='AtomicLeafFreezeTiledController',
        normalized_state_type_paths=STATE_TYPE_PATHS,incremental_reuse_comparison=False,
        incremental_atomic_leaf_freeze_comparison=True,persistent_memory_type_unchanged=True,
        both_controllers_use_tiled_geometry_and_original_cached_clone=True,
        imported_module_globals_mutated=False)
