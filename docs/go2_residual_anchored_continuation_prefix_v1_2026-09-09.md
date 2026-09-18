# Prospective residual-anchored continuation replay V1

This distinct development candidate follows the completed negative hold replay
47f8ab41c12d91d9f46a1def1308218a5eead9ca6c18651683051477001db89a
and completed original-veto diagnosis
14a6b2cc7bcc889823426ab9961444754a849cde6c3d8a60514480db82ee68c8.
The former preserves all 3,004 decisions through the original terminal without
an intervention. The latter reconstructs all 2,643 hold scores and finds that
at 2,619 holds every better movement has a later nominal-path veto unaffected
by a first-point-only correction. Both results and all old failures remain.

## Scientific change

The original no-action recovery and first-point hold recovery retain precedence.
Only if first-point hold recovery returns the original feasible intermediate
WAYPOINT hold may the new candidate act. Given the raw action-conditioned XY
prediction q[a,h] and the existing strictly causal eight-tick first-interval
residual mean b, calculate in the original current body coordinates:

    c[a,0] = q[a,0] - b
    c[a,h] = c[a,0] + (q[a,h] - q[a,0]), h = 1..7

This anchors the model's later relative displacements at its corrected first
point. It changes later prediction positions. The later correction is not
calibrated and is not an error bound. It is an explicit nominal policy exception,
not a clearance certificate. No horizon, radius, contact penalty, action menu,
score weight, model weight, tracker, perception, map, goal or memory changes.
In particular, this candidate does not incorporate the separate maze 1 height
correction or any support-cache performance candidate.

Retain the existing 100 ms causal waypoint utility, full 800 ms contact score,
all original surface vetoes, corrected first-pose surface vetoes, all eight
corrected nominal path segments at radius 0.45 m, current-disk clearance and
phase constraints. No forced movement: only an eligible action with strictly
greater original utility than hold can replace hold. Full original raw forecasts,
scores and veto receipts remain in the decision. Record corrected XY positions,
anchoring equation, residual source ticks, corrected veto receipts and explicit
uncalibrated status. Raw forecasts remain the targets for future residuals.
Yaw and contact predictions stay unchanged. Empty/zero causal correction,
reentry, view exhaustion and final mission-target policies do not activate this
new recovery.

## Prospective evidence boundary

Run scripts/replay_go2_residual_anchored_continuation_prefix_v1.py against exact
completed original residual native result
55a7d5071f39337b3c9ea329e5b48320f11c8a5a9ba6e34296926006768ce466.
Start a fresh controller, map, residual state and original assigned model at
observation zero. Use the same original packets, command completion evidence and
seed/thread settings. Compare full decisions, raw forecasts, public arrays and
all observed/mission/residual state. Stop at the first changed requested command,
before obtaining a following old observation or decision, or at the original
terminal with at most 3,004 observations. Retain a no-change result as negative.
No alternate action outcome is inferred from this replay.

The comparator retains the original first-point receipt checker when that
recovery acts, and separately checks the new full corrected XY construction,
causal bias, utility ranking, eligibility and truthful horizon declarations.
Any unlisted changed decision/state field rejects. A successful comparison is
not proof of navigation; a positive replay requires separately prepared fresh
native execution, full raw audit and exact physical-prefix comparison before
any physical improvement claim. No new native scene is launched by this runner.

## Custody, resources and execution

Exclusive output: go2_residual_anchored_continuation_prefix_v1_attempt_001 under
the navigation development artifact volume. All source dependencies and exact
completed predecessor artifacts are authenticated before output/model replay
and freshly after it. Admit the completed scoped-verifier benchmark
137867773bfe6c6eb05a125a288012ff6017aa3134f4687d7bccdc7f99c02071
before each scope. Original native-verifier conditions remain; every cached file
is freshly rehashed at scope completion and no cache survives the call.

Use the existing Genesis interpreter, PYTHONDONTWRITEBYTECODE=1,
PYTHONHASHSEED=0, PYTHONPATH=.:lewm_genesis:lewm_worlds and one OMP/MKL/OpenBLAS/
OpenCV/PyTorch thread. One CPU replay, no GPU training, no native scene.
Before admission inspect CPU topology/affinity/utilization, RAM, GPU/VRAM,
storage and competing processes. Require 16 GiB replay RAM allowance plus a
conservative 32 GiB concurrent native allowance, 2 GiB total output allowance
(1 GiB compressed stream ceiling), and the standing 40 GiB artifact reserve.
These are admission/planning allowances, not OS-enforced memory quotas.
The existing native height worker continues independently; no running source
or scientific definition changes. No deletion, cleanup approval inference,
sealed access, held-out qualification, physical hardware or deployment authority
is included. The broader goal remains active and unachieved.
