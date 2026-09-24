"""Existing persistent-intent assay with explicit joint sensor ownership."""
from lewm.intent_room_return_development import IntentRoomReturn
from lewm.raw_pulse_runtime_development import RawPulseExecution
from lewm.joint_temporal_anchor_continuity_development import JointTemporalAnchorVisualLedMotion
from lewm.joint_continuous_pulse_execution_development import JointInnerGoalPulseExecution


class RawJointInnerGoalExecution(RawPulseExecution):
    def __init__(self, table, *, identity=(0,0,0), **budgets):
        self.motion=JointTemporalAnchorVisualLedMotion(identity=identity)
        self.executor=JointInnerGoalPulseExecution(table, identity=identity, **budgets)


class JointInnerGoalRoomReturn(IntentRoomReturn):
    def __init__(self, sign, table):
        super().__init__(sign, table, multi_reference=False)
        self.runtime=RawJointInnerGoalExecution(table)
