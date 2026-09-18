# Supervised commitment-contact maze1 native pilot V1

Prospective development-only execution following the completed contact-horizon
causal replay, result SHA-256
`37b29828635e88fab77f81447f6a05b890911426fc3478d8aac451426a229de0`.
Its four observed frames show that contact cost at the actual100ms commitment
selects right arc[.16,0,-.45] instead of right turn[0,0,-.45] at frame3, with
unchanged public observed-state receipts, full model forecasts and feasibility
checks. No physical benefit is inferred from that replay.

Execute one fresh development maze1 case
`full_supervised_commitment_contact_maze_01`, full inputs, supervised-rollout
objective, model `seed_2026091001_full_supervised_rollout`, model state
`171c8576d2c3fcfd0ce698351acf86a05f2ea29bdf829041e98641cdac778a73`.
Keep its exact training-only translation correction. Use the unchanged maze1
specification, public mission, native scene, renderer, robot/gait, friction,
camera configuration,50x2ms physics samples per command,100ms commanded interval,
warmup,3000 shared navigation ticks and10 terminal-zero drain ticks. Keep the
original measured-floor observer/map/mission/memory and view-reentry behavior.

The separately frozen CommitmentContactController changes only intermediate
waypoint contact-cost horizon800ms to100ms. Pose utility remains100ms,
coefficient1.2, all eight800ms nominal path segments and original surface/phase
vetoes remain unchanged. No new observer, contact threshold, prediction repair,
memory reset, final-goal rule, action menu or action-dependent tuning. Contact
scores and pose uncertainty are uncalibrated. Original inputs, model, sources,
diagnostic and complete stopped replay are authenticated before execution.

Runner: `scripts/run_go2_supervised_commitment_contact_maze01_pilot_v1.py`.
Exclusive root: `go2_supervised_commitment_contact_maze01_pilot_v1_attempt_001`
under the guarded external navigation artifact base. One fresh spawned worker,
one native scene, one numerical/OpenCV thread; a second fresh identical model
performs the complete raw sensor/model/command audit. No training or real robot.
New explicit episode/audit sources preserve the original measured-floor native
calculations except controller identity and an honest enabled flag; AST tests
compare every original collection, artifact-enumeration and audit calculation.
No imported frozen module globals are patched.

Before any new native scene, require completion of the current fixed queue with
launch SHA
`651a5815275ecfdd20c5ff6ff7f4d8cdbe4b5c52bbcaf3ad124a6ac97ffd2fb7`:
supervised mazes1–3, direct-flow maze3, anchored-continuation maze2 and recent
qualified-reference maze1, in that exact order. Their original queue must finish
with all original raw-audit and physical-prefix completion receipts, whether
navigation succeeds or fails. Authenticate its completed result and artifact
bindings at the exact result SHA supplied to the runner; bind that identity and
receipt into this new launch and reverify it thereafter. This completion check
does not claim a second execution of the queue's original native input verifiers.
Require no live same-user native runner/spawn worker immediately before new
output creation. This is conservative process observation, not a universal lock.
No edits, skips, retries or added jobs inside the original frozen queue.

Read-only --preflight-only may run while the queue is live: it checks all current
scientific inputs and the original queue launch/source identity, creates no
output and explicitly leaves queue completion unverified. Actual execution also
requires --queue-result-sha256 for the completed original queue; it cannot use a
preflight launch. An absent result or live owner means wait for that same queue,
not a new native attempt. No outcome-based choice of this fixed maze1 case.

Compare fresh and original supervised native physics/public observations through
frame3 only:900 physical samples, four paired camera/body observations, three
identical prior commands, one identical complete raw forecast bank. Require every
new native decision in this prefix to equal the saved prospective replay and
every saved comparison to reconstruct. Require the changed right-arc command to
complete physically. Stop this paired comparison before its differing physical
future; evaluate the complete new episode independently with the unchanged raw
audit and native outcome evaluator. Preserve all raw data and every failure.

Success measurement retains the original outbound and home arrival, quietness,
contact, route/backtracking and strict visibility requirements. No arrival or
round trip is accepted from a prefix or controller declaration alone. A negative
but fully raw-audited episode is still recorded. Process, acquisition, model,
source, verification or prefix failures retain an explicit terminal failure;
there is no automatic retry, resume, tuning or replacement attempt. The native
definition and timing assumptions are not changed: physics remains paused
during computation, so wall timings must be reported without real-time claims.

Before substantial verification and launch assess CPU affinity/utilization,
RAM, GPU/VRAM, competing jobs and both storage volumes. Require32GiB available
native RAM and the original40GiB reserve plus8GiB collection allowance and3GiB
persistence headroom. These are admissions, not enforced resource limits.
Poll resource use while the single worker runs; an observation timeout does not
terminate or restart it. Pre/post original transitive input checks use the
benchmarked isolated scoped digest helper, retaining original conditions and
fresh final hashes without imported-global mutation or persistent cache.

This one reused development layout cannot establish independent-maze reliability,
JEPA/planning/memory advantage, calibrated physical safety, real-time sensing or
deployment readiness. All comparative and independent-layout work in the active
goal remains required. No qualification, promotion or real-platform authority is
implied by this protocol.
