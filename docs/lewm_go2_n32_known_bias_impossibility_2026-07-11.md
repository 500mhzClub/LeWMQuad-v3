# Go2 N32 global known-bias impossibility result

Date: 2026-07-11

Status: fit-only analytical result; no replay, model output, holdout, G2, or
sealed access was required.

## Question

The width-24 N32 V2 model nearly passed fit. Its conditional FREE versus
OCCUPIED confusion at step 2,000 was `[[118763,30],[0,10228]]`: all occupied
cells were ranked OCCUPIED rather than FREE once classified as known. The
remaining joint errors were predominantly at the UNKNOWN versus KNOWN
boundary.

The smallest calibration hypothesis is one scene-blind global known bias:

```text
z_UNKNOWN'  = z_UNKNOWN
z_FREE'     = z_FREE + delta
z_OCCUPIED' = z_OCCUPIED + delta
```

This transformation preserves the FREE/OCCUPIED conditional ordering exactly
and moves only the UNKNOWN/KNOWN boundary.

## Monotonic proof

For every cell and any fixed argmax tie order:

- the indicator that an UNKNOWN target is predicted UNKNOWN is non-increasing
  in `delta`;
- the indicator that an OCCUPIED target is predicted OCCUPIED is
  non-decreasing in `delta`, because the FREE/OCCUPIED ordering is unchanged.

Therefore UNKNOWN recall is non-increasing and OCCUPIED recall is
non-decreasing in `delta` for every aggregate or family panel.

At the three immutable V2 terminal evaluations:

| Step | rough UNKNOWN recall | open-field OCCUPIED recall |
| --- | ---: | ---: |
| 1,800 | 0.973987 | 0.975450 |
| 1,900 | 0.974272 | 0.977087 |
| 2,000 | 0.974518 | 0.978723 |

Both fixed gates are `>= 0.98`.

- Improving rough-terrain UNKNOWN recall from its value at `delta=0` requires
  `delta < 0`; no positive delta can improve it.
- Improving open-field OCCUPIED recall from its value at `delta=0` requires
  `delta > 0`; no negative delta can improve it.
- `delta=0` already fails both gates.

The feasible global-bias interval is therefore empty at every terminal
checkpoint. Balanced-accuracy gates cannot rescue a candidate that already
fails these two mandatory recalls.

## Decision

Global scene-blind UNKNOWN/KNOWN threshold calibration is ruled out without a
retraining replay. Independent review confirmed that fixed argmax ties do not
alter the monotonic argument.

Using separate post-hoc FREE and OCCUPIED biases would alter the already solved
conditional decision and add a higher-dimensional fit-selected calibration;
it is not the tested minimal hypothesis and is not licensed by this result.

The smallest next retrained intervention is an explicit hierarchical output:
one KNOWN logit and one OCCUPIED-given-KNOWN logit, converted analytically to
normalized UNKNOWN/FREE/OCCUPIED log probabilities. This is not more
expressive than three unconstrained logits. Its purpose is to isolate the
UNKNOWN/KNOWN gradient from the solved conditional FREE/OCCUPIED gradient.

If that decoupling fails under the unchanged N32 data, schedule, controls, and
gates, the next representation-level fault is the body-centered polar decoder:
its columns are not true camera rays because the camera is 0.326 m forward of
the body origin. A monotonic horizontal first-hit target is not licensed,
because observable-physical-v3 cells aggregate multiple vertical 3D rays,
ground center/corner evidence, sparse surface witnesses, and collision vetoes.
