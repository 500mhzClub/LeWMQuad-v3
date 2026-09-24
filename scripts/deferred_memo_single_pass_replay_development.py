"""Full paired history with a change only to the footprint receipt copier."""
import builtins
from types import FunctionType,SimpleNamespace
from lewm.single_pass_body_projected_controller_development import SinglePassBodyProjectedController
from lewm.deferred_memo_single_pass_controller_development import DeferredMemoSinglePassController,normalize_to_single_pass
from scripts import single_pass_body_projected_replay_development as previous
from scripts.single_pass_body_projected_state_development import normalized_state_tree,STATE_TYPE_PATHS
from scripts.navigation_artifact_root_development import BASE

SOURCE='scripts/deferred_memo_single_pass_replay_development.py'
TEST='lewm/tests/test_deferred_memo_single_pass_replay_development.py'
OUTPUT=BASE/'go2_deferred_memo_single_pass_late_history_v1_attempt_001'
original=previous.original


def normalize_candidate(decision):
    return previous.normalize_candidate(normalize_to_single_pass(decision))


def progress(*args,**kwargs):
    if args and args[0]=='SCOPED_FOOTPRINT_PAIRED_FRAME':
        args=('DEFERRED_MEMO_SINGLE_PASS_PAIRED_FRAME',*args[1:])
    builtins.print(*args,**kwargs)


def isolated_replay():
    function=original.replay
    view=SimpleNamespace(**vars(original.profile));view.normalize_candidate=previous.normalize_candidate
    namespace=dict(function.__globals__,profile=view,
        FrozenFootprintAnchoredController=SinglePassBodyProjectedController,
        ScopedFootprintAnchoredController=DeferredMemoSinglePassController,
        normalize_candidate=normalize_candidate,state_tree=normalized_state_tree,OUTPUT=OUTPUT,print=progress)
    clone=FunctionType(function.__code__,namespace,function.__name__,function.__defaults__,function.__closure__)
    clone.__kwdefaults__=function.__kwdefaults__
    return clone


def scope():
    return dict(baseline='SinglePassBodyProjectedController',candidate='DeferredMemoSinglePassController',
        normalized_state_type_paths=STATE_TYPE_PATHS,incremental_reuse_comparison=False,
        incremental_deferred_memo_copy_comparison=True,persistent_memory_type_unchanged=True,
        original_packed_insertion_unchanged=True,
        both_controllers_use_original_single_pass_bounds_and_body_projection=True,
        only_two_pure_footprint_copiers_changed=True,imported_module_globals_mutated=False)


def replay(rows,prior_report):
    report=isolated_replay()(rows)
    if report['observed_state_checks'] != prior_report['observed_state_checks']:
        raise ValueError('all seven original completed state identities required')
    return report | scope()
