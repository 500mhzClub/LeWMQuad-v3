std_msgs/Header header
uint32 sequence_id
uint32 block_size
float32 command_dt_s
string primitive_name
float32[] vx_body_mps
float32[] vy_body_mps
float32[] yaw_rate_radps
string event_name
bool event_allowed_in_training
# Collector-policy tag for this block (data spec §5.1 command_source enum):
# "primitive_curriculum", "route_teacher", "frontier", "recovery",
# "ou_noise", "loop_revisit", "teleop", "scripted". Empty string is treated
# as legacy "primitive_curriculum" for backward compatibility.
string command_source
# Privileged label: graph node id of the teacher's current goal cell, or -1
# if no route teacher was active. Labels-only; never an input to the
# deployed model (privileged-leak rule, docs/v3_hjepa_plan.md §3.4).
int32 route_target_id
# Privileged label: graph node id of the next waypoint the teacher was
# steering toward when emitting this block, or -1 if not applicable.
int32 next_waypoint_id
