"""Prospective maze execution bounds; no changes to predecessor attempts."""
from lewm.matched_model_goal_probe_development import WARMUP_TICKS, DRAIN_TICKS
from lewm.observed_round_trip_mission_development import MAX_NAVIGATION_TICKS
from lewm.intent_return_rgbd_replay_development import MAX_FRAMES

NAVIGATION_TICKS = 3000
MAX_COMMAND_TICKS = WARMUP_TICKS+NAVIGATION_TICKS+DRAIN_TICKS
MAX_OBSERVATIONS = MAX_COMMAND_TICKS+1
assert NAVIGATION_TICKS <= MAX_NAVIGATION_TICKS and MAX_OBSERVATIONS <= MAX_FRAMES
RESERVE_BYTES = 40*1024**3
PERSISTENCE_HEADROOM_BYTES = 1024**3
COLLECTION_ALLOWANCE_BYTES = 12*1024**3
