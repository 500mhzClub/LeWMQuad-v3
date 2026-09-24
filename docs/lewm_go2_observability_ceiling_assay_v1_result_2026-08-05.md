# Observability-ceiling assay V1 result

Date: 2026-08-05
Attempt: `go2_observability_ceiling_assay_v1_attempt_v2`
Governing documents: preregistration `..._preregistration_2026-08-05.md`,
`..._amendment_1_2026-08-05.md`, `..._integrity_replacement_v1_2026-08-05.md`

**Registered terminal: `FAIL_ASSAY_CAPACITY_CONTROL`.**
**No Outcome I, II, III, or IV may be claimed, and none is claimed here.**

Result: 324,298 bytes, SHA-256
`c59c8e322e684f4ce5aaa5beeb568365f4a5fc5c8347dfc2026b7d60610ba3c5`.

---

## 1. What the registered rule returned

Amendment 1 required **both** validity controls to pass before any Outcome could
be evaluated. They split:

| control | measure | value | threshold | result |
|---|---|---:|---:|---|
| 2b expressivity | in-sample train regret of the dense visual arm at the top rung | `0.00000` | `<= 0.05` | **pass** |
| 2a identifiability | scene-disjoint regret of an unconstrained MLP on privileged physical successor state | `0.14054` | `<= 0.05` | **fail** |

Because 2a failed, the assay is invalid and the ceiling figure it produced
(`0.30884`) is **not** a registered ceiling. Everything in §3 below is reported
as observation, not as a registered result, and no threshold may be relaxed to
convert it into one.

## 2. What control 2b established, and it is not small

The readout family reaches **exactly zero** in-sample rank regret at the top
rung:

| rung | parameters | train regret |
|---|---:|---:|
| `rung0` | 245 | `0.17540` |
| `rung1` | 6,561 | `0.02058` |
| `rung2` | 99,969 | `0.00000` |

The dense attention readout can express a perfect ranking function over nine
branches from spatially-varying token panels. **Expressivity is not the
bottleneck**, and the frozen 245-parameter interface used by the earlier V-JEPA
physical-interface ceiling was genuinely capacity-limited: it could not fit even
the training panel.

## 3. Observations (not registered Outcomes)

### 3.1 Scene-disjoint regret by arm, at the inner-selected rung

| arm | evaluation regret |
|---|---:|
| `physical_oracle` | `0.00000` |
| `privileged_physical_successor` (bilinear path) | `0.19001` |
| `task_action_only` | `0.30036` |
| `dinov2_true_successor` | `0.30884` |
| `context_only` | `0.34569` |
| `vjepa2_1_true_successor` | `0.35243` |
| `random_expected` | `0.47652` |

The privileged actual-successor visual arm is **worse** than the non-visual
task/action control, and the V-JEPA comparator is worse still, on a panel where
prediction error has been removed entirely.

### 3.2 Capacity buys perfect training fit and no generalization

| arm | rung0 | rung1 | rung2 |
|---|---:|---:|---:|
| `dinov2_true_successor` | `0.28748` | `0.30884` | `0.29740` |
| `vjepa2_1_true_successor` | `0.25784` | `0.35243` | `0.26225` |
| `context_only` | `0.26981` | `0.34569` | `0.30859` |
| `privileged_physical_successor` | `0.19076` | `0.19001` | `0.19001` |

Across a 400-fold parameter range the evaluation regret of the visual arms moves
within noise, while train regret falls to exactly zero. This is a **pure
cross-scene generalization gap**, not an expressivity limit and not prediction
error — the successors here are the actual rendered successors.

### 3.3 Even privileged physics does not reach the gate

The unconstrained MLP on privileged physical successor state — body-frame
displacement, yaw change, path length, fall and tip flags, plus the goal — scored
`0.14054` scene-disjoint. That is above the `0.13` absolute gate that §11 and
§13 both preregistered.

This is precisely the observation the assay was built to obtain, and it arrived
through the control rather than through the primary arm. It must not be treated
as the registered ceiling, because the same number is what failed control 2a.

### 3.4 Power: the decisive quantitative finding

Bootstrap CI half-widths on this 32-scene panel, and the scene count required to
resolve a `0.02` effect at the observed variance:

| comparison | CI half-width | scenes needed for `0.02` |
|---|---:|---:|
| `context_only` − `dinov2_true_successor` | `0.04968` | `197` |
| `dinov2_true_successor` − `privileged_physical_successor` | `0.05318` | `226` |
| `context_only` − `task_action_only` | `0.08954` | `641` |

**Resolving a `0.02` effect on this family needs roughly 200–640 scene clusters.
Every panel used so far has had 16 or 32.** This confirms, with a direct variance
estimate, that the `>= 0.02` relative gates in §11 and §13 were unreachable at
the sample sizes used, independent of the true effect. Any future preregistration
on this family must cite a required scene count before fixing a relative gate.

### 3.5 Branch displacement spread — the action grid is a real contributor

Within-state spread of `physical_target_progress_m` has quartile edges
`[0.00096, 0.04693, 0.08166, 0.11047, 0.19302]` m. A quarter of evaluation states
separate their nine branches by under `4.7` cm, against a `1` cm rank tolerance.

Regret conditioned on that spread:

| quartile | spread range (m) | `dinov2_true_successor` | `context_only` | visual advantage |
|---|---|---:|---:|---:|
| Q0 | `0.001`–`0.047` | `0.36350` | `0.38452` | `+0.021` |
| Q1 | `0.047`–`0.082` | `0.38936` | `0.47221` | `+0.083` |
| Q2 | `0.082`–`0.110` | `0.26153` | `0.31231` | `+0.051` |
| Q3 | `0.110`–`0.193` | `0.22098` | `0.21373` | `-0.007` |

Two things follow, and they point in opposite directions:

- **The action grid genuinely limits the panel.** Regret on the best-separated
  quartile (`0.22098`) is far better than on the two worst
  (`0.36350`, `0.38936`). Widening branch separation would materially improve
  measured regret.
- **It is not sufficient.** Even on the most separated quartile, regret is
  `0.22098` — still well above the `0.13` gate, and there the visual arm no
  longer beats `context_only` at all. A better action grid would move the number
  without reaching the threshold.

This is a concrete, quantified design input for any successor collection, and it
does not depend on the failed validity control.

## 4. Honest reading of the 2a failure

The failure is **ambiguous between two causes** and this assay cannot separate
them:

1. the dense rank genuinely is not recoverable from privileged physical state
   across scenes at this data scale; or
2. the control was under-powered — a learned MLP fit on 128 states and
   generalizing to 128 disjoint states is itself subject to the same
   cross-scene generalization gap seen in §3.2, so it may fail while the target
   remains analytically identifiable.

Cause 2 is plausible: the rank is a deterministic sort over
`(fell, tipped, -quantize(progress), quantize(path))` at a 1 cm tolerance, and
progress is analytically recoverable from body-frame displacement and the goal.
Recovering a 1 cm ordering by *learned* regression is a strictly harder problem
than the algebra suggests.

The correct repair is a **closed-form** identifiability check — compute the rank
directly from the privileged features and confirm zero regret without a learned
model — which removes the generalization confound from the control entirely.
That is a materially different control and **requires its own preregistration**.
It is deliberately not run here, because swapping a failed control for a
more permissive one after seeing the result is exactly the post-hoc manoeuvre
this protocol exists to prevent.

## 5. Custody

The one-way cost declared in preregistration §4 has been paid. Access ledger,
exactly as expected: `train_context` 384, `train_successor` 1,152,
`eval_context` 384, `eval_successor` 1,152. The V3 panel is spent for
privileged-successor purposes.

Integrity gates that did pass: collection rehash to the registered SHA-256, role
disjointness (0 shared scenes), 8×4×4 role balance per role, exact RGB open
counts, `physical_oracle` regret exactly `0.0`, and the primary arm beating
random.

## 6. What follows

No Outcome is claimed, so the `0.13` gate is **neither vindicated nor
overturned** by this attempt, and the dense action-conditioned JEPA successor is
**neither justified nor blocked** by it.

What *is* now on the record, independent of the failed control:

- the readout family can fit the ranking perfectly in-sample (§2);
- it does not transfer across scenes at any capacity tested (§3.2);
- privileged actual-successor vision loses to a non-visual control on this panel
  (§3.1);
- relative gates of `0.02` need an order of magnitude more scenes than any panel
  used so far (§3.4).

The last point is actionable immediately and does not depend on the failed
control.
