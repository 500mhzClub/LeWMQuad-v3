# Independent maze1 failed on correspondence support

The first fixed independent-layout case completed its native collection and
full raw audit. It failed to reach the outbound target, stopping at decision214
with unavailable current visual evidence. The cohort retained that scientific
negative and advanced to maze2. This is the first completed independent-layout
navigation episode under the current controller; it is not a navigation success.

## Audited episode

Root:`/home/andrewknowles/RecoveryStorage/LeWMQuad-v3/.generated/navigation_development_artifacts_v1/go2_independent_floor_transport_mazes_v1_attempt_001`.

| Artifact | SHA-256 |
| --- | --- |
| launch.json | 3053ca602d8e45700550188a3da12e69c3b83314af5a74f32bc616c5425b91c9 |
| full_jepa_novel_maze_01/result.json | b403ca7dd2d15fced064d798c3cd204641289e7eb78c74607e51c51d89cfaf84 |
| full_jepa_novel_maze_01_audit.json | 7719c771ef8b5ecdde828b3e70b0839167be6394b38eb8bbf3f16b0396763b05 |
| full_jepa_novel_maze_01_worker_terminal.json | 97d2075639548de17e73f8857703975b57bebadef6497ea8e3fbe2f59496f1dc |
| cohort_progress_after_01.json | 966e102bbc20bd78e0d3c3ffb90c9e3de00082e42c58ee48ec2d2871ffd1615a |

Collection has225paired observations/decisions,224completed commands and
11,950physics samples, including10terminal zero commands. Physical and
acquisition stops are null. Worker wall time734.5053428560495s; peak
RSS3,303,362,560bytes. All raw sensor reconstruction, model/command replay,
command and unchanged-model checks pass. Strict physical visibility passes
and hard measurement failures are empty. There are no arrival windows and
no verified round trip. Thirteen navigation episodes are now completed/audited,
including one independent-layout case; total verified round trips remain zero.

At213the controller selectedleft_arc`[.16,0,.45]`. At214it commanded zero
with`SENSOR_OR_MODEL_FAILURE` and`same-episode current visual evidence required`.
The retained original visual evidence gives the underlying failure: neither
the primary nor auxiliary camera could register against the previous frame or
any retained anchor206–213. The controller's floor-registration stage never
received a valid current visual pose at214. This differs from maze0's missing
floor extent and subsequent return-ranking stall.

## Completed fixed-frame diagnosis

Script:`scripts/diagnose_go2_independent_maze01_correspondences_v1.py`, SHA-256
`0097924ae70c96a57eb65a690787191c7776a4b297e623daadd1507e8d06e37b`.
Exclusive root:
`/home/andrewknowles/RecoveryStorage/LeWMQuad-v3/.generated/navigation_development_artifacts_v1/go2_independent_maze01_correspondence_diagnosis_v1_attempt_001`.
Result SHA-256:`15ec380fc03906fef61dbd5c9a8418856c63bdf531337152db7d8ad0ac685a56`.
Launch SHA-256:`9d98467e4887832bbea9d06c158d9bed7e56792f6e6642a82c59fb513a1c3241`.

Session83838closedexit0. All1,659source bindings and1,386input artifact
bindings were verified before/after. It reconstructed the actual paired
public images/depth at206–214 with the existing packet reader, detector,
matcher, joint proposal generator and pruning arithmetic. All16recorded
reference rejection reasons reproduced exactly in2.9185069389641285s.
No model, simulator, controller command, pose adoption or threshold change.

| Reference frame | Primary lifted matches | Auxiliary lifted matches |
| --- | --- | --- |
| 206 | 2 | 0 |
| 207 | 4 | 0 |
| 208 | 5 | 1 |
| 209 | 6 | 1 |
| 210 | 4 | 2 |
| 211 | 9 | 3 |
| 212 | 8 | 8 |
| 213 | 10 | 12 |

The unchanged minimum is12matches. For auxiliary reference213,94of129
joint proposals were conditioned enough to fit, but the best proposal had
only11inliers. The recorded error says`insufficient rigid consensus after
pruning`; specifically, the best initial consensus was already below12,
so no iterative pruning/refit round was entered. This clarifies the generic
error wording without changing it.

Frame213contains56primary and76auxiliary selected features; frame214contains
62primary and89auxiliary selected features. All these detected features are
liftable under the existing depth rules. The images are therefore not devoid
of features. Most candidate features do not produce retained cross-frame
correspondences through the unchanged mutual descriptor, duplicate-location,
forward/backward optical-flow and depth-lifting checks. This diagnosis does
not yet separate losses at each association stage or explain the one auxiliary
match excluded by the best rigid proposal.

No gyro value was invented: the reproduced joint proposal/pruning computation
precedes the later gyro gate and does not use gyro. Fraction, grid support,
displacement, gyro consistency and complete pose admission were not evaluated
for a replacement candidate. More raw matches or a hypothetical combined-camera
set alone would not establish a valid pose or successful navigation.

## Resources and next action

Hardware16819closed before the diagnosis with81,586,540,544bytes availableRAM,
94,449,823,744artifact-volume free bytes,all32logical CPUs on16physical cores,
CPU3.3%,bothGPUs0%. The diagnosis's own launch refresh measured80,418,488,320
bytesRAM and94,267,109,376artifactfree,CPU3.4%,bothGPUs0%. It used one OpenCV/
BLAS CPU thread alongside the single active maze2worker. No additional scene.

Continue the unchanged fixed cohort through layouts2and3, preserving all
scientific negatives and stopping on infrastructure failure. Before proposing
a tracking successor, examine where the association pipeline loses measured
support. Any new association or multi-camera registration method needs explicit
prospective source, complete causal prefix evidence, independent raw replay
and fresh native outcomes. Do not lower the existing support gate merely to
admit this failed frame. The JEPA-objective comparison, reactive and planning-
memory work remains separate; no advantage is inferred from this diagnosis.
