# Causal subtrajectory preparation V1

This is a separately scoped development derivation from the completed, raw-audited
24-layout counterfactual corpus. No recollection, fitting, checkpoint selection,
GPU execution or final evaluation is part of this package. Existing source,
labels and role assignments remain unchanged.

## Fixed window definition

Use each actually observed pre-contact decision boundary at branch-relative
times 0, .5, 1, 1.5, 2, 2.5, 3 and 3.5 seconds. Each window has four actual RGB/body/
control packets at current−.3, −.2, −.1 and 0 seconds, all available by the decision.
Do not pad a missing image with a future or terminal image. At time0, all five
siblings use the audited canonical stop-branch context and its past images;
after time0, each window uses its own branch. Each window keeps the original
layout role; repeated windows are not independent layouts.

The prospective action is the existing fixed branch request. Reconstruct its
post-slew applied commands from the **current past applied-command history**.
Only the remaining part of the original 4-second branch is known. Pad tensor
slots beyond this duration with zeros plus an explicit false plan mask; zero
padding is not an executed stop. Do not import future realized feedback or
post-branch release commands. There are eight .5-second horizon slots, with
explicit motion, future-observation and contact validity.

Motion labels are the future-minus-current 3-D base displacement rotated into
the current body frame, keeping its x/y components, and wrapped world-yaw
difference. Compute from original physical records in a separate target builder;
do not subtract earlier 2-D labels, which loses height/tilt information. A motion
or future-image target is valid only at an observed exact timestamp **strictly
before first contact** and within the remaining planned interval. This deliberately
excludes a contact-boundary pose even if recorded. No motion after an emergency
stop is invented. Contact is absorbing within the known interval once observed;
absence is valid only through observed time. Non-contact early termination is
right-censoring, not a safe outcome. Beyond the planned interval every target is
invalid, even after a contact.

The learning-facing loader consumes only policy packets, prospective commands
and separately produced target labels. Its observation inputs contain no world
pose, maze graph, layout identifier, target or future image. Layout/action/role
bookkeeping stays in metadata. Future packets belong only in targets. The
target builder may read the exact corpus's already audited raw development
traces; this does not grant access to historical or sealed material.

## Checks and claims

Freeze this specification and new source before deriving one manifest. Bind the
input result/raw-audit, explicit raw/policy artifacts and output/source identities.
Check role preservation, exact history timestamps, first-contact boundary
semantics, terminal censoring, remaining action coverage and tensor masking with
synthetic fixtures. Independently recompute relative labels using scalar physical
reference equations and compare initial windows against the old labels where
both are valid. Report all counts by role and offset; do not choose offsets from
model scores. No learning outcome will be claimed from a successful data audit.

Later contexts in different action branches are **different physical states**.
Only the initial shared context has five executed counterfactual alternatives.
These windows broaden temporal supervision but cannot on their own establish
new-action coverage, sensor-only replanning success, narrow-maze turning safety,
JEPA benefit or return navigation. A new training/evaluation comparison needs a
separate fixed protocol and matched baselines before any fitting.
