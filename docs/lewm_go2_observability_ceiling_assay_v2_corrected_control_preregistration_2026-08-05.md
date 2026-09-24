# Preregistration: observability-ceiling assay V2, corrected identifiability control

Date: 2026-08-05
Branch: `jepa-spatial-world-model-nav`
Attempt identity: `go2_observability_ceiling_assay_v1_attempt_v3`
Supersedes the validity control of: `..._preregistration_2026-08-05.md` §5.4 as
amended by `..._amendment_1_2026-08-05.md` §4 control 2a

Status: **development-tier repair of a failed validity control on
already-observed arms.** This is explicitly **not** a blind confirmation. §2
states exactly what was already observed before this document was written, and
§3 states why no degree of freedom remains despite that.

---

## 1. Why a new attempt rather than an amendment

Attempt `attempt_v2` returned `FAIL_ASSAY_CAPACITY_CONTROL` and claimed no
Outcome. Its identifiability control 2a — an unconstrained MLP on privileged
physical successor state — scored `0.14054` against the registered `<= 0.05`.

That failure was ambiguous between two causes, and the result record said so:

1. the dense rank is genuinely not recoverable from privileged physical state; or
2. the control was under-powered, because a *learned* model fit on 128 states
   and generalizing to 128 disjoint states is itself subject to the very
   cross-scene generalization gap the assay was measuring.

A control that shares the failure mode under investigation cannot arbitrate it.
The repair is to ask the identifiability question **in principle** rather than
**under finite-sample learning**, which is a materially different control and
therefore requires its own preregistration rather than a second amendment.

## 2. Full disclosure of prior observation

This document is written **after** the following were observed. Recording them
is the point of this section.

1. **All arm values are already known.** `attempt_v2` completed and published
   every arm's scene-disjoint regret, the rung sensitivity table, every paired
   comparison, and both diagnostics. The corrected attempt reuses the same
   immutable collection, the same seeds, and the same deterministic pipeline, so
   the arms will reproduce **identically**. Nothing about the arms is being
   discovered here.
2. **The feasibility of the corrected control is already known.** Before writing
   this document, target progress was checked for closed-form recoverability
   from body-frame displacement and the goal, over all 1,152 evaluation
   branches. Maximum absolute reconstruction error was `6.5e-09` m against the
   `0.01` m rank tolerance — seven orders of magnitude below tolerance.
3. Consequently the corrected control is expected to return exactly `0.0`.

**What this attempt therefore is:** the application of a decision rule that was
frozen before any data was seen, to arm values that are now known, after
repairing a control that failed for a reason internal to itself.

**What it is not:** new evidence about the arms, a blind test, or a
confirmation. Its epistemic status is bounded accordingly in §6.

## 3. Why no degree of freedom remains

Every threshold and every Outcome condition is **inherited unchanged** from the
original preregistration, which was frozen before any run:

- the `0.13` absolute gate;
- the `0.05` validity threshold, applied unchanged to the corrected control;
- the Outcome I / IV / III / II conditions and that evaluation **ordering**;
- all arms, the capacity ladder, the inner split and its seed, the model seeds,
  the bootstrap seed and resample count, the scorer, and the complete-tie
  convention.

Because the thresholds are inherited and the arm values are fixed by determinism,
there is nothing left for the author to tune. The Outcome is a pure function of
already-frozen numbers and an already-frozen rule. Knowing the inputs cannot
change the output.

This is the sole reason the repair is admissible at all. **If any threshold
required adjustment, this attempt would not be legitimate** and the correct move
would be a fresh panel.

## 4. The single change

Amendment-1 control 2a is replaced by a **closed-form** identifiability control:

- reconstruct target progress analytically as `|g| - |g - d|` from the
  body-frame displacement in the privileged feature and the goal carried by the
  state;
- read path length, fall, and tip flags directly from the privileged feature;
- apply the **exact frozen rank rule** of
  `lewm/benchmarks/go2_matched_branch_physical_outcome_screen_v1.py` unchanged —
  key `(fell, tipped, -quantize(progress), quantize(path))`, one-centimetre
  quantization, dense ranking over the sorted distinct keys;
- score by that reconstructed rank and evaluate under the unchanged scorer.

It has no parameters, no training set, and no fitting step, so it carries **no
cross-scene generalization confound**. It asks only whether the dense rank *is*
a function of privileged successor physical state.

Registered threshold, inherited unchanged: **evaluation regret `<= 0.05`.**

Amendment-1 control 2b (expressivity: in-sample train regret of the primary
dense visual arm at the top rung, `<= 0.05`) is retained exactly as registered.
Both controls must still pass for any Outcome to be claimed.

## 5. Interpretation rule, fixed here

The corrected control separates the two causes that `attempt_v2` conflated:

- **If it returns `<= 0.05`**, the dense rank is identifiable from privileged
  physical successor state, and the `attempt_v2` 2a failure at `0.14054` is
  attributed **entirely to finite-sample learning**, not to an unlearnable
  target. That attribution is itself a reportable finding: it means even
  analytically sufficient privileged physics fails to *generalize* across
  disjoint scenes at this data scale, which corroborates the cross-scene
  generalization gap observed in the arms.
- **If it returns `> 0.05`**, the privileged feature set is insufficient to
  express the rank, the assay fails closed again, and no Outcome may be claimed.

## 6. Bounded standing of whatever Outcome results

Because §2 applies, the resulting Outcome:

- **is** a registered application of a pre-frozen rule, and may be cited as the
  assay's terminal;
- **is not** blind, held-out, or confirmatory evidence, and may not be described
  as such;
- **does not** promote any checkpoint, authorize any successor experiment,
  reinstate any stopped mechanism, or license planner integration;
- **remains** development-tier and non-citable as qualification evidence.

Any Outcome that would otherwise read as a licence to proceed must be reported
together with this section.

## 7. Custody

**No additional custody cost.** The evaluation successor RGB was already opened
by `attempt_v1` and `attempt_v2`; the declared one-way cost has been paid and is
not paid again. The V3 panel remains spent for privileged-successor purposes and
its status is unchanged by this attempt. No untouched, sealed, held-out, or V4
material is opened.

## 8. Integrity gates

Unchanged from the original preregistration §8: collection rehash to
`711b8722c11dbae663ad1b004268b77c64ff3d2e818f2c895851c547240e3ed0`, role
disjointness, 8×4×4 role balance, exact RGB open counts, `physical_oracle` regret
exactly `0.0`, the primary arm beating random, exclusive output write, and no
overwrite, resume, or repair of an existing attempt.

Additionally registered for this attempt: the corrected control's reconstructed
rank matrix must equal the collection's own dense rank matrix **exactly** on all
128 evaluation states. A mismatch means the reconstruction is not the frozen rank
rule and the attempt fails closed.

## 9. What this does not authorize

No data generation, rendering, training of any navigation or promotion
candidate, threshold relaxation, further attempt beyond `attempt_v3`, retry of a
*scientific* failure, planner integration, deployment, or any access to
untouched, sealed, held-out, or V4 material.
