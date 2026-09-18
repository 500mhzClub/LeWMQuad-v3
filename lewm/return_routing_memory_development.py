"""Remove accumulated routing evidence only during the return mission leg."""
from lewm.selected_route_turn_memory_development import SelectedRouteTurnMemoryMixin
from scripts.run_go2_sparse_corner_completion_development import CompletionRuntime


class ReturnRoutingMemoryRuntime(SelectedRouteTurnMemoryMixin, CompletionRuntime):
    return_scope = 'persistent'

    @property
    def routing_memory_scope(self):
        # The existing mission controller freezes this generation for a plan
        # and discards its commands if the mission changes before dispatch.
        generation = getattr(self, 'planning_generation', None)
        return self.return_scope if generation is not None and generation > 0 else 'persistent'

    def _select_action(self, *args, **kwargs):
        selected, correction = super()._select_action(*args, **kwargs)
        return selected | dict(return_routing_memory_treatment=dict(
            planned_mission_generation=self.planning_generation,
            return_scope=self.return_scope, outbound_scope='persistent',
            scope_changes_only_after_observed_goal=True,
            accumulated_action_clearance_preserved=True)), correction


class CurrentPairReturnRuntime(ReturnRoutingMemoryRuntime):
    return_scope = 'latest_mapped_pair'


RUNTIMES = dict(persistent_return=ReturnRoutingMemoryRuntime,
    current_pair_return=CurrentPairReturnRuntime)
