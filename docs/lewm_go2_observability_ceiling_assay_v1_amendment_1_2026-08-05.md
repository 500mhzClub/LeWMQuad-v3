# Amendment 1 to the observability-ceiling assay V1 preregistration

Date: 2026-08-05
Amends: `docs/lewm_go2_observability_ceiling_assay_v1_preregistration_2026-08-05.md`
Attempt identity: `go2_observability_ceiling_assay_v1_attempt_v1` (not yet consumed)

Status: **pre-result mechanism correction.** This amendment changes the
*mechanism* of the assay-validity control in preregistration §5.4. It changes
**no registered threshold, no Outcome condition, no Outcome ordering, no arm,
no seed, no scorer, and no gate value.** Every number in §5.5 and §6 of the
preregistration stands exactly as frozen.

---

## 1. The defect

Preregistration §5.4 defined the assay-validity control as arm 2,
`privileged_physical_successor`, whose 6-dimensional physical feature is embedded
into a dense panel by zero-padded replication across all 256 patch slots so that
it travels through the *same* readout family as the visual arms.

That construction is degenerate. In the readout,

```
hidden_i      = tanh(W_r r_i + W_p p_i + W_q c + b_h)
attention     = softmax_i(w_alpha . hidden_i)
values_i      = W_v r_i
pooled_value  = sum_i attention_i * values_i
score         = w_z . pooled_value + pooled_value^T B c + b_score
```

When every patch carries the identical feature `r_i = r`, the values are constant
across patches, so

```
pooled_value = sum_i attention_i * W_v r = W_v r
```

exactly, because the attention weights sum to one. The attention still *varies*
across patches — the positional term `W_p p_i` differs — but it **cancels
completely**. The score therefore reduces to

```
score = w_z . (W_v r) + (W_v r)^T B c + b_score
```

which is **bilinear in the feature and the condition, with no nonlinearity at
all**. The tanh is inert on this path.

This was verified directly: with identical tokens, attention standard deviation
is nonzero while `pooled_value == W_v r` to `1.19e-07`.

Consequently arm 2 as specified did not test what §5.4 claimed it tested. It
measured whether a *bilinear* function of the privileged feature can rank the
branches — not whether the readout family, operating on genuinely
spatially-varying dense panels, can express a ranking function.

## 2. The observation that exposed it

A pre-run validity check of the control at the registered 256-epoch schedule
returned:

| rung | parameters | privileged-arm evaluation regret |
|---|---:|---:|
| `K=8, H=4` | 245 | `0.19076` |
| `K=32, H=32` | 6,561 | `0.19001` |

A 27-fold parameter increase moved the result by `0.00075`. That flatness is the
signature of the degeneracy: both rungs collapse to the same bilinear model
class, so capacity is irrelevant on this path. Under §5.4 this would have emitted
`FAIL_ASSAY_CAPACITY_CONTROL` and forbidden every Outcome — correctly failing
closed, but for a reason internal to the control rather than to the science.

## 3. Disclosure of prior observation

Full disclosure, recorded because it is material to the integrity of the assay:

- An **infrastructure smoke run at `EPOCHS = 2`** with two rungs and no V-JEPA
  arm was executed before this amendment, to verify wiring, custody accounting,
  and shapes. It printed evaluation regret for the primary arm
  (`dinov2_true_successor` `0.283`), `context_only` (`0.301`), and
  `task_action_only` (`0.300`) at that degenerate two-epoch budget.
- The registered protocol is 256 epochs. Those two-epoch numbers are not
  registered results and are not usable as evidence.
- **Every threshold, Outcome condition, and Outcome ordering in §5.4, §5.5 and
  §6 was frozen before any run and is unchanged by this amendment.** The smoke
  observation therefore cannot have influenced any decision boundary.
- The smoke run wrote no result to the registered output path.

## 4. The replacement control

Preregistration §5.4 is replaced by **two** controls. Both must pass or the assay
is invalid, `FAIL_ASSAY_CAPACITY_CONTROL` is emitted, and no Outcome in §6 may be
claimed. The registered threshold `0.05` is retained unchanged for both.

**Control 2a — identifiability of the target.** An unconstrained
three-layer MLP (hidden width 128, GELU) on the concatenation of the privileged
6-dimensional physical successor feature and the 4-dimensional condition,
trained on the same schedule, seeds, split, and residual objective. This
establishes that the dense rank *is* a learnable function of the successor
physical state, i.e. that the target and labels are sound.

Registered threshold: **evaluation regret `<= 0.05`.**

**Control 2b — expressivity of the readout family on dense panels.** The
*in-sample* regret of the `dinov2_true_successor` arm at the top rung, evaluated
on the train states it was fit on. This travels the genuine
spatially-varying dense path, where attention does not cancel and the tanh is
live. It establishes that the readout family can express a ranking function from
dense token panels.

Registered threshold: **train-set regret `<= 0.05`.**

Together these separate the two failure modes the original single control
conflated: an unlearnable target, and an inexpressive readout.

## 5. Status of arm 2

`privileged_physical_successor` is **retained as a reported arm** and its
evaluation regret is still published, but it is **relabelled** from
"capacity/identifiability control" to what it actually is: a *bilinear
privileged-feature reference*. It is no longer a validity gate, and it no longer
appears in the Outcome III condition of §6.

Because §6's Outcome III previously referenced
`dinov2_true_successor` versus `privileged_physical_successor`, that clause is
narrowed to its remaining, unchanged condition: Outcome III holds when the
visual ceiling fails to beat the non-visual `task_action_only` control. No
threshold changes; one degenerate comparator is removed from a disjunction.

## 6. What is unchanged

The immutable inputs, custody declaration, arms 1 and 3 through 7, objective,
residual formulation, capacity ladder, inner scene-disjoint split and its seed,
refit procedure, scorer, complete-tie convention, bootstrap seed and resample
count, model seeds, the `0.13` absolute gate, the `0.05` validity threshold, the
Outcome I/IV/III/II ordering, all mandatory diagnostics, and all integrity gates.
