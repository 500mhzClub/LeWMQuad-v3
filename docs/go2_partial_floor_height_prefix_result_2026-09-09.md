# Completed partial-height command-boundary replay

Session 20949 exited 0, result
b8338e4954569251e6052356a6b74b3056a6190fc971eb5445125fe3d179a374,
root go2_partial_floor_height_prefix_v1_attempt_001. Independent check 49620
revalidated all 1,693 source bindings and both output bindings, and the native
prefix admission re-compared all saved decisions against the original tracking
episode. All checks pass. Wall time after admission, including replay and final
ancestry verification: 1,165.182160672 s.

All 504 preceding complete decisions are exact after normalizing only the new
controller label and enable flag. The 501 preceding forecast banks and actual
commands match. Raw visual tracker output remains exact throughout; public
arrays and the assigned model state remain unchanged, with no gradients.

At frame 504, the original floor conflict is repaired by a height increment of
−0.0010598303276462828 m along the retained initial floor normal. The total
position correction relative to raw visual pose is 0.010495008198278028 m,
within the original 5 cm gate. Rotation and normal remain transported; the full
anchor stays at frame 490. All 5,956 auxiliary candidates remain represented;
corrected maximum residual 0.002332356675147418 m, RMS
0.0010500088445653242 m. The primary camera has zero current floor candidates.
No partial plane is promoted into a full anchor or physical identity claim.

The full controller recovers with no terminal or failure and selects left arc
[0.16, 0, 0.45] instead of the original zero request. It stops at this first
changed command on observation 504; observation 505 is never consumed.
This establishes a prospective command-level intervention, not the outcome
of that unexecuted command. Fresh physical collection and the complete
raw/physical prefix audit remain necessary.

Bindings:

- launch.json: 9db1c429ee00d5b0e5d5c44117c57eb1a0b49ca1c6df57ff4205751357629df3
- context_decisions.jsonl.gz: 0ae0f230292db829cbee8ee09e10c6f407f6c9bc088804ca4646eb04242702c8

Hardware at independent admission: 75,247,927,296 RAM bytes available,
75,622,105,088 artifact bytes free, 21,358,596,096 workspace bytes free; CPU
3.3%, 16 physical/32 logical/all affinity, GPUs idle. Only hold replay PID
2509963 remained as a competing development Python job, RSS 7,663,939,584 bytes.
No native scene was running. Aggregate remains 22 completed/raw-audited native
episodes and zero verified round trips.
