# Preregistration: separated-branch panel V1

Date: 2026-08-05
Branch: `jepa-spatial-world-model-nav`
Attempt identity: `go2_separated_branch_panel_v1_attempt_v1`
Answers the registered next step of:
`docs/lewm_go2_observability_ceiling_assay_v2_corrected_control_result_2026-08-05.md` §7

Status: **development-tier data collection plus a blind re-run of the frozen
observability-ceiling assay.** Unlike the V2 corrected-control attempt, this is a
**genuinely blind test**: the panel does not yet exist, so no arm value is known
to the author at preregistration time. It opens no untouched, sealed, held-out,
or V4 material.

---

## 1. Why this experiment, and why it is the only admissible next step

The observability-ceiling assay reached a valid registered terminal of
`OUTCOME_IV_PANEL_DEGENERATE`. Its registered meaning:

> The branch construction does not produce visually distinguishable outcomes;
> the action grid or step length must be redesigned before any successor
> collection.

That Outcome forbids another readout, encoder, capacity, or threshold change on
the V3 panel, and it names exactly two permitted levers: **branch separation**
and **scene count**. This preregistration changes those two things and nothing
else.

Both requirements are quantified rather than guessed:

- **Separation.** Regret by within-state displacement-spread quartile was
  `0.36350 / 0.38936 / 0.26153 / 0.22098`. The panel's lower half is where the
  ceiling is worst, and a quarter of states separate all nine branches by under
  `4.7` cm against a `1` cm rank tolerance.
- **Scenes.** Measured CI half-widths of `0.04968`–`0.08954` on 32 scene
  clusters imply `197`–`641` clusters to resolve a `0.02` effect.

## 2. The two changes

### 2.1 Branch separation: action grid V2

The V3 grid contains three near-duplicate forward speeds — `0.30`, `0.25`, `0.20`
m/s — over a five-step block. Those three branches are the dominant source of
unresolvable ties. Two minimal changes, neither of which leaves the locomotion
policy's trained velocity envelope:

| change | V3 | V2 |
|---|---|---|
| candidate block length | 5 steps | **10 steps** |
| forward speed triple | `0.30 / 0.25 / 0.20` | **`0.30 / 0.20 / 0.10`** |

All other commands are **unchanged**: arcs at `vx 0.20, wz ±0.45`, backward
`-0.20`, hold, and turns-in-place at `wz ±0.45`. Peak commanded velocity is
unchanged at `0.30` m/s and `0.45` rad/s, so no action leaves the envelope the
frozen controller was trained on.

Together these multiply the forward triple's mutual separation by roughly four
and double every branch's displacement.

History blocks remain 2, at their existing length, so the context contract is
untouched.

**Registered separation check, evaluated before any model is fit:** the median
within-state displacement spread must exceed `0.11` m — the V3 top-quartile edge.
If it does not, the grid change failed its purpose, the attempt terminates as
`FAIL_SEPARATION_NOT_ACHIEVED`, and no assay is run.

### 2.2 Scene count: 304 scenes, 152 per role

Selection reuses the frozen V3 contract — lowest fresh SHA-256 ranks per family,
role allocation by role hash, 4 states per scene — with the count raised from 8
to **38 scenes per family**. The fresh cap is 39 — set by the smallest eligible
families, `rough_local_dynamics` and `visual_sensor_stress`, which have 39 fresh
each after the V3 panel consumed 8 — and 38 is the largest **even** value below
it, so the existing role-allocation rule splits each family exactly in half.

| quantity | V3 | V2 |
|---|---:|---:|
| scenes | 64 | **304** |
| scenes per role | 32 | **152** |
| states per role | 128 | **608** |
| candidate branches | 2,304 | **10,944** |
| RGB frames | 3,072 | **14,592** |

**Expected power at 152 clusters**, scaling the measured half-widths by
`sqrt(32/152) = 0.4588`:

| comparison | V3 half-width | expected V2 |
|---|---:|---:|
| `context_only − ceiling` | `0.04968` | `0.02279` |
| `ceiling − task_action_only` | `0.05781` | `0.02653` |

At the V3 effect size, `context_only − ceiling` would become approximately
`[+0.01406, +0.05964]` — **excluding zero**. Outcome IV would therefore no longer
fire on the same effect, and the assay would reach Outcome III or II. That is the
specific reason this panel size is sufficient to make progress.

**Disclosed shortfall.** 152 clusters gives a half-width of about `0.0228`,
which is still ~14% short of the `0.02` needed for the registered relative gates,
requiring `197`. A balanced fresh panel cannot reach 197 without either reusing
consumed scenes or unbalancing the families, and both were rejected. If a
relative gate fails only because its interval spans zero at this width, that
result must be reported as **underpowered, not negative** — the same error §14.2
identified in §11 and §13.

### 2.3 Family-yield rule, fixed in advance

`rough_local_dynamics` is known to fail the rigid solver on some scenes and to be
skipped cleanly by the pipeline. Collect 38 per family; if any family yields
fewer than 35 usable scenes, **all** families truncate to the minimum achieved,
by the frozen selection rank order, and the truncation is reported with the
result. Family balance is never broken to preserve a target count.

## 3. What is inherited unchanged

Everything not named in §2:

- the `0.13` absolute gate and the `0.05` validity threshold;
- the Outcome I / IV / III / II conditions and that evaluation ordering;
- both validity controls — closed-form identifiability 2a and expressivity 2b;
- all seven arms, the three-rung capacity ladder, the residual-on-task-ridge
  objective, the inner scene-disjoint split rule, the refit procedure;
- the scorer, the `max(1, max_dense_rank)` denominator, and the complete-tie
  convention;
- the family-balanced whole-scene bootstrap at 10,000 resamples;
- every integrity gate, including exact RGB open counts and role disjointness.

Seeds are drawn fresh for a fresh panel and are fixed here: collection seed
`20260805`, split seed `2026080553`, model seeds
`(2026080561, 2026080562, 2026080563)`, bootstrap seed `2026080552`.

## 4. Blindness

No arm value on this panel is known, and the panel does not exist at
preregistration time. This attempt therefore **restores the confirmatory
standing** that the V2 corrected-control attempt explicitly lacked, and its
Outcome may be reported as a blind development result — still development-tier,
still non-citable as qualification evidence, but not subject to the bounded
standing clause of that attempt.

To preserve blindness, no intermediate arm score is inspected before the
registered evaluation completes. The only quantity examined between collection
and evaluation is the §2.1 separation check, which is a property of the physics
and carries no model score.

## 5. Custody

A **fresh** panel with a fresh successor role. The V3 panel remains spent and is
not reused, reopened, or referenced as evidence. The new evaluation successor RGB
is opened once by the assay, which is that panel's own declared one-way cost.

No untouched, sealed, held-out, or V4 material is opened. This is not the sealed
V4 benchmark and does not approach it.

## 6. Registered terminals

- `FAIL_SEPARATION_NOT_ACHIEVED` — the §2.1 median-spread check fails; no assay.
- `FAIL_INFRASTRUCTURE_NO_SCIENTIFIC_DECISION` — collection or evaluation stops
  before a decision; no Outcome.
- `FAIL_ASSAY_CAPACITY_CONTROL` — either validity control fails; no Outcome.
- Otherwise the assay's registered Outcome I, IV, III, II, or
  `INCONCLUSIVE_NO_REGISTERED_OUTCOME`, unchanged.

## 7. What this does not authorize

No promotion, no planner integration, no closed-loop or navigation claim, no
threshold relaxation, no retry of a scientific failure, no deployment, and no
access to untouched, sealed, held-out, or V4 material. A pass on any Outcome
determines only what gate a separately preregistered successor experiment must
face.
