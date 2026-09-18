"""Same observed settling in comparator collection and complete raw replay.

Model assignments, independent scenes and the population schedule belong to
the prospective experiment. Importing this module executes no simulation.
"""
from lewm.stop_conditioned_comparators_development import CONTROLLERS
from scripts import extended_return_budget_comparator_pipeline_development as previous

bind = previous.extended.bind
functions = bind(previous.functions, CONTROLLERS=CONTROLLERS)
execute = bind(previous._execute, CONTROLLERS=CONTROLLERS, functions=functions)
collect = bind(previous.collect, _execute=execute)
audit = bind(previous.audit, _execute=execute)
artifacts = previous.artifacts
resource_artifacts = previous.resource_artifacts
