"""Fixed predictor, reactive, and local route-turn memory comparisons."""
from lewm.interrupted_route_turn_memory_development import InterruptedRouteTurnMemoryMixin
from lewm.persistent_visual_baselines_development import (
    ComputedForecastsUnusedMixin, ReactiveFeedbackMixin)
from scripts.run_go2_sparse_corner_completion_development import CompletionRuntime


class NoRouteTurnMemoryRuntime(CompletionRuntime):
    def _select_clear_prediction(self, selected, prediction, snapshot, position, rotation):
        # Skip only the local interrupted-turn memory selector. The existing
        # forecast feasibility, recovery, arrival and stopping chain follows it.
        return super(InterruptedRouteTurnMemoryMixin, self)._select_clear_prediction(
            selected, prediction, snapshot, position, rotation)


class ReactiveCompletionRuntime(ComputedForecastsUnusedMixin,
        ReactiveFeedbackMixin, CompletionRuntime):
    pass


RUNTIMES = dict(jepa=CompletionRuntime, supervised_rollout=CompletionRuntime,
    command_history=CompletionRuntime, reactive_feedback=ReactiveCompletionRuntime,
    jepa_no_route_turn_memory=NoRouteTurnMemoryRuntime)
