# Contact-scoring worker: raw visual failure and association probe

The contact-scoring native worker completed collection and raw audit without
reaching the goal or completing a round trip. Its parent is still performing
final input verification; these results authenticate the completed worker and
do not certify the parent or waiter as complete.

Worker terminal SHA-256:
`5b2791c83053ea944db281bb4e6990e5eddee0a00ac89e4e03ccef8cd185bac3`.
Native launch SHA-256:
`8d6a736c2f1f6de1f461fa6952341c1a7f3d321ccdd37881ac27a6da35191b8c`.
The original worker PID 2813548 is absent on the recorded boot. Its artifact
roster, raw collection, audit, prefix receipt, readout and log were verified
before and after diagnosis. The 572-observation decision stream was read in
full and every executed requested command matched its original tape. The
native contact/timing readout was reconstructed.

The run crossed five distinct outbound edges, had no native contact samples,
passed strict physical visibility, and stopped at observation 561 with
SENSOR_OR_MODEL_FAILURE. Observation 560 is the last admitted pose and decision.
Ten subsequent observations retained terminal zero commands. There were no
arrivals. Median full iteration was 1127.958 ms; all 572 observations exceeded
the 100 ms command interval. The failure is a loss of visual pose support,
not the floor-correction rejection observed in the hold-reorientation run.

Both cameras exhausted eight retained references and the previous-frame
measurement. Every one of the 18 recorded pre-gyro registration failures was
reproduced from the original raw RGB/depth data with unchanged feature,
matching and rigid-registration rules. For the immediately preceding frame:

| Camera | Original valid depth pairs | Original rigid support | Failure |
| --- | ---: | ---: | --- |
| Primary | 6 | Not fitted | Fewer than 12 matches |
| Auxiliary | 12 | 11 consensus points | Fewer than 12 consensus points |

Raw diagnosis:
`docs/go2_contact_anchored_worker_tracking_failure_diagnosis_2026-09-11.json`,
SHA-256 `9f28b43e90a9e4076b61ced5bac7daa43476a2b18d3e6acec8db8ec926244cea`.
Checker `scripts/diagnose_go2_contact_anchored_worker_tracking_failure_v1.py`
completed in session 79031, exit 0, with 1988 bound source paths. The existing
tested register-local tracing implementation was reused with only the actual
frame changed to 561, retaining the original deterministic proposal seed.
Identity gyro was unused before the reproduced failures; the actual gyro gate
and full observer/model history were not reexecuted.

The unchanged direct-corner optical-flow fallback was then tested only on the
same two raw frames, 560 and 561. It produced 10 primary and 21 auxiliary valid
depth pairs. Both association receipts and all four output arrays per camera
were byte-exact on repetition. No threshold was tuned. The primary population
remains below the minimum. The auxiliary population warrants checking the
existing fallback against the complete original tracking history; a larger
match population alone does not establish a qualified pose.

Pair probe:
`docs/go2_contact_anchored_direct_flow_pair_probe_2026-09-11.json`,
SHA-256 `568bd6f7b13428ccd96291eb9945ec33dd72591a79549a7a4b0f841fc388848b`.
Script `scripts/probe_go2_contact_anchored_direct_flow_pair_v1.py` completed in
session 36550, exit 0, with 1989 source paths. This probe did not evaluate rigid
registration, gyro or temporal continuity, admit a pose, select a command,
perform model inference or execute physics.

Next: after the current sustained raw replay releases the CPU replay slot,
reconstruct the original and existing direct-flow observers through frame 561
from the actual public packets. Require the entire original visual history to
match and preserve all qualified original witnesses as conflict vetoes. Only
if the fallback produces a fully qualified current pose should a separately
checked controller prefix and prospective native follow-up be considered.
Do not merge this tracking change into any already frozen or queued run.

Latest process check: raw replay PID 2813368 remained active at approximately
41 minutes elapsed, with frame 300 observed. Contact parent PID 2808232 and
contact/tracking/budget waiters remained active; no final contact or raw result
existed, and the tracking native output was absent. Sustained native waiter
PID 2817601 remained idle behind the original replay and queue. The overall
goal remains active; no new navigation success or policy selection is claimed.
