# Fresh physical maze 1 partial-height pilot V1

This development experiment tests the physical consequence of the scalar-height
intervention defined in go2_partial_floor_height_prefix_v1_2026-09-09.md. It
reuses tracking maze 1 and does not count as an independent generalization
trial. Use PartialHeightDirectFlowController, the same full JEPA model and its
training-only bias, cameras, gyro, gait, dynamics, mission and seed as completed
tracking native result
d6774bae22cb9effeb0cd85ae255de203de1539541f701d57788b58ab00769de.
No model training, performance-cache substitution or hold-policy intervention.

Before execution require a completed, fully authenticated
go2_partial_floor_height_prefix_v1_attempt_001 result. Its SHA-256 is supplied
explicitly to the launcher and retained in the launch and final result. Never
infer admission from a progress message or an incomplete output directory.
The admission checker re-compares every saved replay decision against the
original tracking episode: 505 observations, exact decisions and actual requests
before frame 504, 501 preceding raw prediction banks, current admitted partial
height, full-controller recovery and a changed nonzero selected command at 504.
All live-pose validation must have passed in the completed replay. Unchanged
hold, negative terminal, truncated, failed, altered-input or altered-model
replays do not admit this native attempt. Do not select a different command
after looking at a physical outcome.

Fresh collection begins at the initial episode state. Retain 3 warmup ticks,
3,000 navigation ticks shared between outbound and return, and 10 terminal zero
commands. One CPU native scene, one spawned task, one numerical/OpenCV thread;
physics pauses during computation. The original collector and raw audit differ
only in controller class and collection status labels. Tests compare their ASTs
against the frozen tracking predecessor. Every new complete decision identifies
the partial-height controller. Native pose and topology remain evaluator inputs
only; no simulator truth becomes a controller observation.

The independent raw audit loads a fresh identical assigned model, reconstructs
all primary and auxiliary camera observations and complete decisions, verifies
all requested/completed commands and unchanged model weights, and applies the
same strict visibility, measurement, native settled arrival, route-retracing,
contact, terminal-quiet and physical/acquisition-stop evaluation. The candidate
has no calibrated pose or physical-floor-identity guarantee. A component/replay
pass or observed mission arrival is not a verified round trip.

Physical-prefix comparison covers 25,950 samples through observation 504,
505 paired public packets, all 504 preceding actual commands and all 505 complete
prospective candidate decisions. Require the current command at 504 to equal
the prospectively admitted request. Compare no physics or observations after
that command starts. A partially completed changed command remains a recorded
negative physical outcome; it does not erase a valid shared past. Independently
audit the entire fresh episode to assess what actually happens after the change.
Preserve collected artifacts and completed audits if later comparison fails.

Output is exclusive go2_partial_floor_height_maze01_pilot_v1_attempt_001. Run
scripts/run_go2_partial_floor_height_maze01_pilot_v1.py with
--prefix-result-sha256 set to the completed exact replay result. First execute
--preflight-only: it must return before output creation or worker/model/scene
execution. Authenticate source, predecessor, replay, model and native bindings
before launch and again at completion. Any infrastructure or verification error
retains files and stops this attempt; no automatic retry or in-place resume.

Require 32 GiB available RAM and the standing 40 GiB artifact reserve plus
10 GiB collection allowance and 1 GiB persistence headroom. These are admission
checks, not enforced OS quotas. Immediately before each substantial launch,
refresh CPU topology/affinity/load, RAM, GPU/VRAM, storage and competing jobs.
Preserve one native scene at a time; bounded CPU analyses may overlap only when
the complete active allowances fit measured headroom.

Scheduling: this new corrective experiment may occupy an idle native slot once
its completed positive replay and resource checks pass. The previously prepared
supervised fixed cohort retains its unchanged three-case definition and storage
gate; the prepared tracking maze 3 retains its own declared queue dependency.
This protocol neither shrinks that cohort nor edits its frozen launch rules.
No pending cleanup approval is inferred, no deletion is included, and no running
attempt is interrupted. If another native scene is active, wait for it to end.

No sealed access, real-robot action, production/promotion or deployment follows.
A positive repeated-layout result alone would not establish independent-maze
reliability, JEPA or memory advantage, 100 ms operation or hardware readiness.
