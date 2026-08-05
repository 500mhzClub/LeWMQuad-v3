# Observability-ceiling assay V2 result: corrected identifiability control

Date: 2026-08-05
Attempt: `go2_observability_ceiling_assay_v1_attempt_v3`
Governing document:
`docs/lewm_go2_observability_ceiling_assay_v2_corrected_control_preregistration_2026-08-05.md`

**Registered terminal: `OUTCOME_IV_PANEL_DEGENERATE`. The assay is valid.**

Result: 324,689 bytes, SHA-256
`f08dbcf957a976483e3a9135a155d5a59e55c5e6cf2865a2c51eea7850c0d7d2`.

**Read §5 before citing this. The Outcome is a registered application of a
pre-frozen rule to arm values that were already known. It is not blind,
held-out, or confirmatory evidence.**

---

## 1. Both validity controls now pass

| control | form | value | threshold | result |
|---|---|---:|---:|---|
| 2a identifiability | **closed form** (V2 correction) | `0.00000` | `<= 0.05` | **pass** |
| 2b expressivity | in-sample train regret, top rung | `0.00000` | `<= 0.05` | **pass** |

The closed-form control reconstructed the collection's dense rank matrix
**exactly** on all 128 evaluation states — the additionally registered integrity
gate of preregistration §8. Target progress reconstructs from body-frame
displacement and the goal to a maximum absolute error of `6.5e-09` m against the
`0.01` m rank tolerance, seven orders of magnitude below tolerance.

## 2. The `attempt_v2` failure is now explained, and the explanation is a finding

`attempt_v2` failed control 2a at `0.14054` with a *learned* MLP on the same
privileged physical successor state that the closed-form control now maps to
regret `0.0`.

The target was therefore **perfectly identifiable all along**. The learned
control failed because it shares the exact failure mode the assay was built to
measure: it is a finite-sample model asked to generalize across disjoint scenes.

This is not merely a cleared obstacle. It is corroborating evidence for the
central observation:

> Even privileged physical state that is **analytically sufficient** to
> reconstruct the rank exactly cannot be **learned** to reproduce that rank
> across disjoint scenes at this data scale.

A control contaminated by the effect under investigation cannot arbitrate it —
and when replaced by an uncontaminated one, the contamination itself measured
`0.14054`.

## 3. Determinism check

All seven arms reproduced **bit-for-bit** between `attempt_v2` and `attempt_v3`:

| arm | regret (identical in both attempts) |
|---|---:|
| `physical_oracle` | `0.00000000000000000` |
| `privileged_physical_successor` | `0.19001116071428570` |
| `task_action_only` | `0.30036272321428570` |
| `dinov2_true_successor` | `0.30884486607142850` |
| `context_only` | `0.34569382440476193` |
| `vjepa2_1_true_successor` | `0.35242745535714280` |
| `random_expected` | `0.47651703042328040` |

This confirms the pipeline is deterministic and that the corrected attempt
changed only the validity control, exactly as preregistered.

## 4. The Outcome, and the condition that did not fire

Evaluation follows the pre-declared order I → IV → III → II; the first condition
that holds is the terminal.

| Outcome | condition | held? |
|---|---|---|
| I gate achievable | `R* <= 0.13` | no — `R*` is `0.30884` |
| **IV panel degenerate** | `context_only − ceiling` interval includes zero | **yes** — `+0.03685`, CI `[-0.01363, +0.08572]` |
| III no visual headroom | `ceiling >= task` | **also holds** — `0.30884 >= 0.30036` |
| II gate too tight | ceiling securely beats task | no |

**Both IV and III conditions hold.** The pre-declared ordering selects IV, and
IV is the registered terminal. Reporting only the one that fired would be
misleading, so both are recorded: the panel is degenerate *and* the visual
ceiling does not beat the non-visual control. They are consistent, not
competing — a panel whose branches are not visually distinguishable is precisely
one on which vision cannot beat a non-visual baseline.

The registered meaning of Outcome IV, unchanged from the original
preregistration:

> The branch construction does not produce visually distinguishable outcomes;
> the action grid or step length must be redesigned before any successor
> collection.

## 5. Bounded standing — required reporting

Per preregistration §6, this Outcome:

- **is** a registered application of a rule frozen before any data was seen, and
  may be cited as the assay's terminal;
- **is not** blind, held-out, or confirmatory evidence. The arm values were
  already published by `attempt_v2` and the pipeline is deterministic, so they
  were known before this attempt was preregistered. The feasibility of the
  corrected control was also known;
- **is admissible only because** every threshold and the Outcome ordering were
  inherited unchanged from the original preregistration, leaving no free
  parameter to tune. Had any threshold required adjustment, a fresh panel would
  have been required instead;
- **does not** promote any checkpoint, authorize the dense action-conditioned
  JEPA successor, reinstate any stopped mechanism, or license planner
  integration;
- **remains** development-tier and non-citable as qualification evidence.

## 6. What now stands on the record

With the assay valid, the observations recorded in the `attempt_v2` result are
promoted from unregistered observation to findings of a valid assay:

1. **The readout family can fit the ranking perfectly in-sample** — train regret
   `0.17540 → 0.02058 → 0.00000` across 245, 6,561 and 99,969 parameters.
   Expressivity is not the bottleneck.
2. **It does not transfer across scenes at any capacity tested.** Evaluation
   regret for `dinov2_true_successor` moves within noise
   (`0.28748 / 0.30884 / 0.29740`) across a 400-fold parameter range while train
   regret reaches exactly zero. With actual successors supplied, this is a pure
   cross-scene generalization gap — neither expressivity nor prediction error.
3. **Privileged actual-successor vision loses to a non-visual control**:
   `0.30884` and `0.35243` against `task_action_only` `0.30036`.
4. **Relative gates of `0.02` need `197`–`641` scene clusters** at the measured
   CI half-widths of `0.04968`–`0.08954`. Panels used to date have had 16 or 32.
5. **The action grid is a real but insufficient limiter.** Regret by
   displacement-spread quartile runs `0.36350 / 0.38936 / 0.26153 / 0.22098`;
   even the best-separated quartile sits well above `0.13`, and there the visual
   arm no longer beats `context_only` at all.

## 7. What this does and does not settle about the `0.13` gate

The registered terminal is Outcome IV, **not** Outcome I or II. So the assay
does **not** report an achievable ceiling for the `0.13` gate, and the gate is
**still neither vindicated nor overturned**.

What it reports instead is that this panel cannot answer the question, because
its branches are not visually distinguishable enough for actual successors to
add measurable information over context alone. The correct next step is a panel
with wider branch separation and materially more scene clusters — §6.4 and §6.5
quantify both requirements — not another readout, encoder, capacity, or
threshold change on this panel.

## 8. Custody

No additional custody cost. The evaluation successor RGB was already opened by
`attempt_v1` and `attempt_v2`; the declared one-way cost was paid there and is
not paid again. Access ledger for this attempt was exact:
`train_context` 384, `train_successor` 1,152, `eval_context` 384,
`eval_successor` 1,152. No untouched, sealed, held-out, or V4 material was
opened.
