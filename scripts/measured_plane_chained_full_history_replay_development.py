"""Private copies of the complete paired replay for the chained controllers.

No launcher, input admission, queue or native execution is performed here.
The future caller must authenticate the completed chained native result and
create its exclusive output before invoking these replay/check functions.
"""
from types import SimpleNamespace

from lewm.measured_plane_chained_anchor_development import MeasuredPlaneChainedAnchorController
from lewm.measured_plane_chained_single_pass_controller_development import MeasuredPlaneChainedSinglePassController
from scripts import measured_plane_chained_full_history_timing_development as comparison
from scripts import replay_go2_measured_plane_single_pass_full_history_v1 as previous
from scripts import run_go2_measured_plane_chained_maze02_v1 as native
from scripts.extended_budget_anchored_maze_development import _bind


def adapters(output):
    """Retain the original paired loop and checker without changing globals."""
    original = SimpleNamespace(CASE=native.CASE, assigned_model=native.assigned_model,
        state_digest=previous.original.state_digest,
        public_mission=previous.original.public_mission,
        ArticulatedCollisionGeometry=previous.original.ArticulatedCollisionGeometry)
    common = dict(OUTPUT=output, native=native, original=original,
        comparison=comparison, packets=previous.packets)
    replay = _bind(previous.replay, **common,
        MeasuredPlaneResidualController=MeasuredPlaneChainedAnchorController,
        MeasuredPlaneSinglePassController=MeasuredPlaneChainedSinglePassController)
    check = _bind(previous.check_output, **common)
    return replay, check


def replay(admission, *, output):
    return adapters(output)[0](admission)


def check_output(report, admission, *, output):
    return adapters(output)[1](report, admission)
