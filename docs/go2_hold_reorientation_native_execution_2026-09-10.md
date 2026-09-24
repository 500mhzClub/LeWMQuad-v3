# Hold-reorientation native launcher and automatic handoff

Snapshot: 2026-09-10 10:39 UTC. The preceding turn completed source preparation
and physical-prefix checks. This turn implemented and tested the single-case
native launcher, completed-input admission, and automatic handoff. The goal
remains active: 37 completed audited development episodes, zero verified round
trips. No new native case has launched during this work.

The new waiter is PID **2671835**, created **1789036731.41**, tool session
**88131**. Command:
`.generated/venvs/genesis_rocm_0_4_6_v1/bin/python scripts/await_go2_hold_reorientation_maze02_native_v1.py`.
Output: `go2_hold_reorientation_maze02_native_wait_v1_attempt_001` under the
external navigation artifact root. Launch SHA-256:
`e27675df102b072b62f4351483363ab9ee80e9e0244e1193c398392f716a47f6`.
All 1,960 bound sources are now frozen. The current event is
`WAITING_FOR_RAW_REPLAY_AND_SCHEDULED_FRONTIER`, with both original owners live.

The existing order is preserved:

1. Adapter batch PID 2659758 completes its six original cases. First JEPA worker
   PID 2662101 is still running audit/finalization; no worker audit/terminal has
   yet completed. Its collection previously stopped after 1,529 observations
   and 1,528 completed commands with no feasible candidate.
2. CPU waiter PID 2669739 launches the full hold-reorientation raw replay once
   that exact first worker finishes and validates. It may run alongside a later
   original batch case. It owns the raw replay; do not launch another manually.
3. Existing frontier waiter PID 2663938 launches its already specified maze 03
   native experiment after all six adapter cases complete and validate.
4. New native waiter PID 2671835 waits for both the CPU replay waiter and frontier
   waiter to end with verified complete results. It then launches exactly one
   `run_go2_hold_reorientation_maze02_pilot_v1.py` child using their pinned hashes.

The new child requires the same original first worker to be bound into the
completed six-case batch. It reconstructs both completed handoffs, reexecutes
full original input admission for the raw and frontier chains, verifies the
raw prefix from saved comparisons and public packets, then requires an idle
native slot. All original sources and artifacts remain fixed. Scientific
navigation failure is an admissible completed result; incomplete or changed
evidence stops the handoff. No retry, replacement, source mutation or restart
is allowed. Waits poll the exact boot/PID/creation/command identities every
30 seconds for up to 48 hours.

The planned case is `full_jepa_hold_reorientation_maze_02`, with the unchanged
expanded full-JEPA adapter model and the separately tested hold-reorientation
controller. It retains the original scene, mission, sensor timing, robot, gait,
navigation budget and full raw audit. The paired comparison requires 21,000
matching preintervention physics samples, 406 matching public observations,
identical completed commands through 404, and actual completion of the new
left-turn request at 405 through physics sample 21,049. Later outcomes are
measured independently. The frontier policy is not merged into this candidate.

Validation:

- Final launcher/input-chain suite: session 85454, 22 passed in 2.07 seconds.
  Includes acceptance of audited scientific failures and rejection of changed
  model assignments, incomplete physical boundaries, false success, substituted
  first workers and mismatched completed batches.
- Native waiter suite: session 35713, five passed in 2.12 seconds. Covers both
  original owners, pinning the first completion, missing evidence, changed
  identity and bounded waiting without replacement.
- Final source-only preflight: session 58005, 1,957 sources, approximately
  70 GiB available RAM and 606 GiB artifact space. No runtime output created;
  completed-input admission was not performed by that preflight.
- After registration, the new native waiter's 1,960 sources, CPU waiter's 1,946,
  frontier waiter's 1,929 and original batch's 1,908 were independently
  reverified unchanged. All five recorded parent/worker/waiter processes were
  confirmed live.

At this snapshot neither `go2_hold_reorientation_maze02_prefix_v1_attempt_001`
nor `go2_hold_reorientation_maze02_pilot_v1_attempt_001` exists. The raw replay
and physical outcome are still pending. No independent comparison layout was
rendered, observed or navigated by this preparation. Queuing the experiment,
passing source tests or completing a turn does not establish navigation,
backtracking, predictive-planning advantage, real-time performance or hardware
readiness. Inspect the exact running owners and their original outputs on the
next continuation; do not restart quiet audits or queued work.
