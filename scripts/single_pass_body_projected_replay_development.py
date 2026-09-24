"""Full original paired loop; change only the candidate's bounds queries."""
import builtins
from types import FunctionType, SimpleNamespace

from lewm.body_projected_tiled_controller_development import BodyProjectedTiledController
from lewm.single_pass_body_projected_controller_development import (
    SinglePassBodyProjectedController, normalize_to_body_projected)
from scripts import body_projected_tiled_replay_development as previous
from scripts.single_pass_body_projected_state_development import normalized_state_tree, STATE_TYPE_PATHS
from scripts.navigation_artifact_root_development import BASE

SOURCE = 'scripts/single_pass_body_projected_replay_development.py'
TEST = 'lewm/tests/test_single_pass_body_projected_replay_development.py'
OUTPUT = BASE/'go2_single_pass_body_projected_late_history_v1_attempt_001'
original = previous.original


def normalize_candidate(decision):
    return previous.normalize_candidate(normalize_to_body_projected(decision))


def progress(*args, **kwargs):
    if args and args[0] == 'SCOPED_FOOTPRINT_PAIRED_FRAME':
        args = ('SINGLE_PASS_BODY_PROJECTED_PAIRED_FRAME', *args[1:])
    builtins.print(*args, **kwargs)


def isolated_replay():
    function = original.replay
    view = SimpleNamespace(**vars(original.profile))
    view.normalize_candidate = previous.normalize_candidate
    namespace = dict(function.__globals__, profile=view,
        FrozenFootprintAnchoredController=BodyProjectedTiledController,
        ScopedFootprintAnchoredController=SinglePassBodyProjectedController,
        normalize_candidate=normalize_candidate, state_tree=normalized_state_tree,
        OUTPUT=OUTPUT, print=progress)
    clone = FunctionType(function.__code__, namespace, function.__name__,
        function.__defaults__, function.__closure__)
    clone.__kwdefaults__ = function.__kwdefaults__
    return clone


def replay(rows, prior_report):
    report = isolated_replay()(rows)
    if report['observed_state_checks'] != prior_report['observed_state_checks']:
        raise ValueError('all seven completed body-projected state identities required')
    return report | dict(baseline='BodyProjectedTiledController',
        candidate='SinglePassBodyProjectedController', normalized_state_type_paths=STATE_TYPE_PATHS,
        incremental_reuse_comparison=False, incremental_single_pass_bounds_comparison=True,
        persistent_memory_type_unchanged=True, original_packed_insertion_unchanged=True,
        both_controllers_use_original_body_projection_and_receipt_handling=True,
        imported_module_globals_mutated=False)
