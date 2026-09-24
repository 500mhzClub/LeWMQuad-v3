# Existing-pool world-model interpretation correction — 2026-08-01

Status: **durable interpretation correction, not execution authority**. This
document corrects the July 31 world-model handoff from first principles. It
does not authorize data access, packing, GPU work, training, evaluation,
checkpoint creation, retry, resume, promotion, G2-G8 work, held-out access, or
sealed access.

Original companion experiment contract:
`docs/lewm_go2_world_model_existing_pool_three_arm_v1_preregistration_2026-08-01.md`.
After two consumed pretraining source failures, the science-identical successful
lineage terminated as integrity replacement V3. Its durable result records are:

- `docs/lewm_go2_world_model_existing_pool_three_arm_v1_integrity_replacement_v3_terminal_review_2026-08-01.json`; and
- `docs/lewm_go2_world_model_existing_pool_three_arm_v1_integrity_replacement_v3_terminal_handoff_2026-08-01.md`.

## Terminal disposition — 2026-08-01

The corrected existing-pool experiment has now run. V3 completed all 700
updates for each arm, and its external receipt checker passed. The registered
scientific decision is `LOCALIZE_ACTION_ALIGNMENT_FAILURE`.

The aligned conditioned arm beat both the candidate-blind and action-shuffled
controls on training fit and scene-disjoint validation. Its balanced action
identification was `0.246934` against `0.111111` chance. This confirms the core
correction below: the existing observational pool contains learnable factual
requested-action signal, and new simulator data was not the immediate critical
path.

The full positive claim did not pass. The hardest-wrong-action margin was
`−0.009454` with one-sided lower quantile `−0.011383`; the later
predictor-usefulness gate was not reached and would also fail against
persistence. The result therefore establishes neither uniform action alignment
nor architecture sufficiency. V3 does not justify bulk world-model data
generation for this factual-action localization. The next bounded diagnosis is
a separately authorized, read-only action-level extraction allowlisting only
the bound update-700 conditioned snapshot and exact bound validation-index
metadata. A requested-versus-executed provenance audit is separate and applies
only if a complete exact role-bound join already exists; otherwise provenance
must be recorded as unavailable before any new training or rendering decision.

## 1. Corrected bottom line

The existing non-held-out Go2 pool is sufficient to run the next decisive
experiment: a scene-disjoint test of whether a frozen visual substrate can
learn **factual requested-action-conditioned** next-state prediction.

No new simulator data is required for that question.

The claim that the on-policy H6 objective contains no action-learning signal
because each exact continuous state appears with only one action is withdrawn.
Exact-state uniqueness prevents direct same-state counterfactual identification;
it does not remove ordinary supervised information about
`target | history, requested action` across many related states and scenes.

The corrected critical path is therefore:

1. audit the existing role's observational support and leakage boundaries;
2. train matched factual, candidate-blind, and action-deranged arms on the
   existing pool;
3. evaluate all arms on the same disjoint factual validation scenes; and
4. generate a small counterfactual evaluation role only if factual
   action-conditioned learning is first demonstrated and an untaken-action
   claim remains necessary.

## 2. What the repository already has

The active non-held-out train/validation pool is approximately 3 TB:

- 2.896 TB decimal / 2.634 TiB allocated across the active raw-rollout and
  textured-RGB roots;
- 55.2 million source-frame rows;
- 10,614,345 sliding reset-safe H6 windows;
- 1,807,552 row-disjoint packed H6 candidates;
- 1,000 train scenes and 150 scene-disjoint validation scenes in the frozen
  selected roles;
- all nine requested action IDs at every visible action position; and
- all 81 adjacent ordered action pairs in the selected train role.

The immediately usable frozen schedule is much smaller than the physical pool:
16,000 train H6 rows and 2,048 validation H6 rows. Those selected rows cover
all 1,150 role scenes, 0.998477% of the available row-disjoint packed H6
candidates, and 0.222618% of the train/validation textured-RGB allocation. This
distinction matters:

- the repository does **not** need another large corpus before testing factual
  action learnability; and
- a negative result on the selected schedule still would not prove that the
  remaining pool lacks useful data.

The physical-footprint and exposure basis is recorded in
`docs/lewm_go2_main_pool_physical_footprint_and_exposure_audit_2026-07-28.md`.

## 3. Why exact-state uniqueness is not a no-signal proof

Robotic visual states are continuous and high-dimensional. Requiring the same
pixel-perfect state to recur under multiple actions is not a prerequisite for
ordinary supervised prediction. A model can learn an action-dependent
conditional from variation across nearby histories, scenes, obstacle layouts,
incoming velocities, and requested actions, provided the observational support
is adequate and the action is not perfectly confounded with those covariates.

Exact repeated states would be valuable because they permit a direct paired
counterfactual comparison. Their absence changes the strength of the claim:

- existing on-policy data can test **factual generalization**;
- it cannot by itself validate predictions for actions not taken from the
  identical physical state; and
- it cannot, without controls, distinguish use of the requested action from a
  scene, policy, or state shortcut.

That is why the next experiment requires both a candidate-blind arm and a
within-family action-deranged arm. Those controls test whether aligned candidate
actions improve factual prediction beyond the same visual histories and action
marginals. They are a more direct test of the alleged missing signal than the
raw exact-state census.

## 4. Existing evidence that requested actions carry physical information

The repository's metadata-only action/frame alignment audit already found
requested-action signal on disjoint validation scenes. On the corrected
boundary-to-boundary transition, requested primitive separability was well
above balanced chance, and requested forward/yaw commands correlated strongly
with realized motion.

Those diagnostics do not prove that the temporal JEPA will learn or use the
signal. They do refute the stronger proposition that requested actions are
physically uninformative in the existing role.

The relevant distinction is:

- **signal present in the observations** is an empirical property already
  supported by the alignment audit;
- **signal learned by this objective and frozen representation** is the open
  question addressed by the three-arm experiment; and
- **untaken-action causal validity** remains a later, separate question.

The action/frame evidence and its causal boundary correction are recorded in
`docs/lewm_go2_main_pool_action_frame_alignment_audit_2026-07-28.md`.

## 5. Requested versus executed command provenance

The corrected H6 role exposes requested action IDs aligned to causal five-tick
boundaries. The selected frame metadata does not provide a complete,
role-bound executed/clipped command tape for those transitions.

Requested actions are the correct deployment-available candidate input: a
planner knows what it proposes before the future occurs. Feeding a future
executed or clipped result as though it were known at candidate-scoring time
would leak downstream controller behavior.

The missing executed-command join nevertheless limits interpretation:

- a requested action may be clipped, delayed, or altered by the controller;
- two requested primitives may induce more similar realized motion than their
  labels suggest;
- the model is learning the closed-loop response of observation history,
  requested command, controller, and environment, not an unconstrained actuator
  oracle; and
- a factual requested-action result cannot establish executed-command fidelity
  or prove that every requested class created a distinct physical intervention.

The new experiment therefore conditions on requested actions, reports this
provenance limitation, and makes no executed-command or untaken-action claim.
If executed-command provenance is later recovered under a separate reviewed
role, it is first an audit and stratification variable, not automatically a
future model input.

## 6. Corrections to the July 31 handoff

### 6.1 Withdrawn: counterfactual generation is the sole critical path

The July 31 handoff concluded that new counterfactual data generation was the
critical path because scaling on-policy H6 creates no within-state contrast.
The factual premise is true; the exclusive conclusion is not.

Within-state contrast is required for a direct untaken-action causal test. It
is not required to test whether factual requested actions improve prediction
on scene-disjoint observational data. The existing pool must be tried first.

### 6.2 Withdrawn: exact-state uniqueness means the objective never teaches
action dependence

One action per exact continuous state does not imply zero conditional signal.
The appropriate first test is a controlled observational comparison, with
scene-disjoint evaluation and a deranged-action baseline.

### 6.3 Retained with a narrower meaning: Diagnostic B is a capacity result

The corrected counterfactual overfit diagnostic showed that the model family
could fit its small training construction better when given aligned actions.
That remains a training-set capacity observation only. It says nothing decisive
about:

- scene-disjoint generalization;
- the validity of the historical malformed counterfactual role;
- optimization at the registered scale;
- factual usefulness on the main pool; or
- predictions for untaken actions.

It neither proves architectural sufficiency nor justifies architectural
escalation.

### 6.4 Retained: direct same-state counterfactual evidence is still missing

The existing pool does not provide all actions from the identical pre-action
physical state. A later paired simulator role may still be required to validate
untaken-action rankings. That need is conditional on first showing that the
factual model learns a useful requested-action signal.

## 7. Disposition of the V3 counterfactual smoke

The V3 smoke remains a valid, bounded source-integration result for its own
collector contract. It established synchronized branching, exact sentinel
behavior, distinct executed candidate tapes, rendering reliability, and bounded
runtime without retrying the consumed V1 or V2 attempts.

It does not change the order of the learning experiments:

- it is not a training corpus;
- it is not evidence that the existing factual pool is inadequate;
- it does not establish generalization;
- its training-set or infrastructure observations are non-citable; and
- its consumed one-shot root must never be reused or refilled.

The smoke makes a later small counterfactual evaluation role more credible to
collect. It does not make that collection the next required operation.

## 8. What the next experiment can and cannot decide

The preregistered three-arm experiment can decide whether, under one frozen
encoder/target substrate and one fixed 700-update protocol, aligned requested
candidate actions improve factual next-state prediction on the 150 disjoint
validation scenes relative to:

- the same model with the candidate-action embedding removed; and
- the same model trained with within-family, marginal-preserving wrong
  candidate actions.

A pass supports the statement:

> The existing selected pool contains learnable factual requested-action
> information for this frozen temporal-predictor protocol.

It does not support any of the following statements:

- the model predicts untaken actions correctly from the same physical state;
- the full 3 TB pool has been trained or exhausted;
- the encoder should be promoted or remains physically qualified;
- the architecture is generally sufficient for world modeling;
- the model is causally used by a deployed planner;
- the world-model subsystem passes WM-A, WM-I, or formal G2-G8; or
- new data will never be required.

A failure supports only a protocol-local statement. Depending on the registered
localization, the cause may remain training fit, generalization/confounding,
action alignment, predictor usefulness, observational support, optimization,
or frozen-feature adequacy. It must not be rewritten as proof that the corpus
contains no action signal.

## 9. Standing decision

The immediate experiment uses the existing selected pool and creates no new
simulator scenes or counterfactual branches.

If factual requested-action learnability passes, the next data operation should
be a small, scene-disjoint counterfactual **evaluation** role, not another
multi-terabyte training corpus. If it fails, follow the preregistered
localization before deciding between objective, optimizer, representation,
observational-support, or data remedies.

Until a separately reviewed authority exists, this correction permits only
source, document, and synthetic-fixture work.
