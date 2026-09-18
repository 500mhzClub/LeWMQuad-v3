"""Revised-perception independent-maze collection and complete raw audit.

Source preparation only: no launcher or population execution authority. Keep
the original independent scenes, private evaluator and physical loop. Compose
the separately defined four-arm assignments with the existing 4,000-tick sensor,
stream and renderer bounds without modifying either predecessor module.
"""
from lewm import measured_plane_independent_round_trip_study_development as study
from scripts import extended_budget_anchored_maze_development as extended
from scripts import independent_round_trip_adapter_multiarm_episode_development as episode
from scripts import independent_round_trip_adapter_multiarm_audit_development as raw_audit
from scripts import reactive_nominal_maze_command_audit_development as reactive_commands
from scripts.independent_round_trip_session_development import IndependentRoundTripSession
from scripts.measured_plane_independent_controller_factory_development import create


class MeasuredPlaneIndependentSession(IndependentRoundTripSession, extended.ExtendedBudgetRendererSession):
    """Retain independent physical initialization and extended paired capture."""


assert study.NAVIGATION_TICKS == extended.NAVIGATION_TICKS
assert study.MAX_COMMAND_TICKS == extended.MAX_COMMAND_TICKS
assert study.MAX_OBSERVATIONS == extended.MAX_OBSERVATIONS
assert study.COLLECTION_ALLOWANCE_BYTES == extended.COLLECTION_ALLOWANCE_BYTES

collect = extended._bind(episode.collect,
    require_case=study.require_case, treatment=study.treatment, verify_model=study.verify_model,
    command_role=study.command_role, COLLECTION_STATUS=study.COLLECTION_STATUS, create=create,
    NAVIGATION_TICKS=study.NAVIGATION_TICKS, MAX_OBSERVATIONS=study.MAX_OBSERVATIONS,
    COLLECTION_ALLOWANCE_BYTES=study.COLLECTION_ALLOWANCE_BYTES,
    writer=extended.writer, IndependentRoundTripSession=MeasuredPlaneIndependentSession)

audit_reactive_commands = extended._bind(reactive_commands.audit_commands,
    NAVIGATION_TICKS=study.NAVIGATION_TICKS)
audit = extended._bind(raw_audit.audit,
    require_case=study.require_case, require_collection=study.require_collection,
    treatment=study.treatment, verify_model=study.verify_model,
    replay_receipt=study.replay_receipt, create=create,
    NAVIGATION_TICKS=study.NAVIGATION_TICKS, MAX_COMMAND_TICKS=study.MAX_COMMAND_TICKS,
    IntentReturnRGBDReplay=extended.ExtendedBudgetRGBDReplay,
    audit_sensors=extended.audit_sensors,
    audit_learned_commands=extended.audit_commands,
    audit_reactive_commands=audit_reactive_commands,
    read_rows=extended.read_rows, packet=extended.rgb_packet,
    renderer_audit=extended.renderer_audit)


def artifacts(case, result):
    # The predecessor imports its old contract inside artifacts(), so this
    # wrapper must explicitly validate the new assignment before enumerating.
    study.require_collection(case, result)
    names = [episode.DECISIONS if name == 'context_decisions.json' else name
        for name in episode.primary_artifacts('', result)]
    return names+[episode.RENDERER_WITNESSES, 'public_mission.json',
        'auxiliary_camera_audit.json', 'decision_stream_timing.jsonl']+[
        f'auxiliary_{kind}_{index:04d}.{suffix}'
        for index in range(result['auxiliary_frames'])
        for kind, suffix in (('depth', 'npz'), ('rgb', 'png'))]
