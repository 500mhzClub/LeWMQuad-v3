# JEPA Navigation Literature and Experimental-History Critical Review

**Review date:** 2026-08-02
**Scope:** Current JEPA and JEPA-world-model literature, navigation-specific evidence, and a first-principles evaluation of this repository's approach and experimental history.

## Verdict

As of 2 August 2026, the repository has produced a credible negative result for its current globally pooled, one-step LeWM/SIGReg training branch. It has **not** established that counterfactual-data generation is the critical path, that architecture is exonerated, or that observational data cannot train a useful action-conditioned JEPA.

The session handoff is unusually honest and scientifically valuable, especially its withdrawn-claims section. Its bottom line nevertheless needs one more correction:

> Counterfactual branches are currently the critical path to a decisive causal **evaluation**, not yet the proven critical path to **training**.

The project is one calibrated branch set plus one controlled training comparison away from a much cleaner causal diagnosis. It remains substantially further away from evidence that the model is useful for navigation planning.

## What the current literature actually says

| Area | Best evidence | Implication here |
|---|---|---|
| Representation JEPA | [I-JEPA](https://openaccess.thecvf.com/content/CVPR2023/papers/Assran_Self-Supervised_Learning_From_Images_With_a_Joint-Embedding_Predictive_Architecture_CVPR_2023_paper.pdf), [V-JEPA](https://arxiv.org/abs/2404.08471), and [V-JEPA 2.1](https://arxiv.org/abs/2603.14482) establish strong abstract and increasingly dense spatiotemporal representations. | Representation quality, rank, and anti-collapse health do not establish controllable dynamics. Dense spatial tokens increasingly appear preferable to one global embedding. |
| Offline latent planning | [DINO-WM, ICML 2025](https://proceedings.mlr.press/v267/zhou25t.html) learns an action-conditioned predictor over frozen DINOv2 patch features and plans with CEM in six simulated task families. [PLDM, NeurIPS 2025](https://proceedings.neurips.cc/paper_files/paper/2025/hash/3e7cf447f21cd11c846463affefce665-Abstract-Conference.html) also learns planning latents from ordinary reward-free offline transitions. | Exact same-state/multiple-action examples are not universally necessary. Local state-action coverage and appropriate inductive bias can be sufficient. |
| Physical JEPA planning | [V-JEPA 2-AC](https://arxiv.org/abs/2506.09985) combines internet-video pretraining with under 62 hours of robot interaction and closed-loop image-goal planning on Franka arms. | Large passive data helps the encoder, but a comparatively small action-labelled interaction set trains the dynamics. Evidence is short-horizon and based on small physical trial counts. |
| Best current design study | The accepted TMLR study [What Drives Success in Physical Planning with JEPA-WMs?](https://arxiv.org/abs/2512.24497) favors dense DINO features, proprioception, AdaLN-style action conditioning, short multi-step training, and CEM. It also finds prediction loss can correlate poorly with planning success. | The repository's global pooling, missing or limited belief state, one-step target, and proxy-centric evaluation are all live failure causes. Model scale alone is not the answer. |
| End-to-end JEPA | [LeWorldModel](https://arxiv.org/abs/2603.19312) is the closest published design to this repository: global latent, next-embedding prediction, SIGReg, and CEM. | It is a useful baseline, not an established recipe for complex navigation. Its own results and follow-up work show sensitivity to task dimensionality, data diversity, horizon, and latent geometry. |
| Action sensitivity | Recent preprints [Delta-JEPA](https://arxiv.org/abs/2606.31232), [ActSWM](https://arxiv.org/abs/2607.26712), and [DWM](https://arxiv.org/abs/2607.18715) improve action-sensitive planning using displacement-action decoding, rollout separation, multi-step losses, or explicit action/world decomposition, all with factual logged transitions. | These directly contradict an objective-independent claim that matched counterfactual successors are the only route forward. They are promising but very recent and not yet strong consensus evidence. |
| Planning geometry and horizon | [Temporal Straightening, ICML 2026](https://arxiv.org/abs/2603.12231), [Predictive but Not Plannable](https://arxiv.org/abs/2605.07278), [Temporal-Distance JEPA](https://arxiv.org/abs/2607.25337), and [Hierarchical Planning with Latent World Models](https://arxiv.org/abs/2604.03208) target curvature, reachability, temporal cost, and hierarchy. | Even a predictive, action-sensitive model can be useless to CEM if Euclidean latent distance has the wrong geometry or the planning horizon is misaligned. |
| Navigation | [PiJEPA](https://openaccess.thecvf.com/content/CVPR2026W/WDFM-EAI/html/Chahe_Policy-Guided_World_Model_Planning_for_Language-Conditioned_Visual_Navigation_CVPRW_2026_paper.html) is relevant but evaluates trajectory errors rather than a robust closed-loop robot deployment; its policy prior can partially bypass weak world-model rollouts. [Navigation World Models](https://openaccess.thecvf.com/content/CVPR2025/html/Bar_Navigation_World_Models_CVPR_2025_paper.html) is generative rather than JEPA and also emphasizes offline prediction/ranking. | Direct JEPA navigation evidence remains thin. This review found no convincing JEPA result for closed-loop Go2 or comparable legged navigation with the world model causally controlling behavior. |

For perspective, [DayDreamer](https://proceedings.mlr.press/v205/wu23c.html) already demonstrated actual physical world-model learning on a quadruped, arms, and wheeled navigation. It is online reward-driven Dreamer rather than reward-free JEPA, but it sets a much stronger evidence bar than latent probes.

## What the repository has actually established

The governing plan asks for untaken-action utility (WM-A), optional rollout composability (WM-C), physical/spatial retention (WM-S), and causal planner use (WM-D); see [the action-conditioned world-model plan](lewm_go2_action_conditioned_world_model_plan_2026-07-31.md).

| Property | Current status | Correct interpretation |
|---|---:|---|
| WM-A | Unmeasured | Existing metrics compare alternative action predictions to the one factual successor. They do not measure whether an untaken action predicts its true successor or reduces physical regret. |
| WM-C | Failed only in the unchanged interface | The predictor consumes 256 lattice tokens but emits 64 masked tokens, so direct re-entry is impossible. Adapters and different rollout architectures remain untested. |
| WM-S | Unmeasured functionally | Freezing the encoder prevented encoder drift; it did not prove that the trained dynamics retain usable spatial/physical structure. |
| WM-D | Absent | No learned world model has yet supplied planner scores that causally determined navigation actions. |

The strongest experiment is the three-arm V3 comparison:

- 16,000 training rows and 2,048 validation rows from 150 scene-disjoint scenes.
- The conditioned model beat the blind and cross-scene shuffled models.
- Balanced action identification reached 0.2469 versus 0.111 chance.
- Yet hardest-action energy ordering was negative, and the model was substantially worse than persistence.

Those results establish **scene-disjoint factual action association**, not causal action-conditioned dynamics. See [the three-arm terminal review](lewm_go2_world_model_existing_pool_three_arm_v1_integrity_replacement_v3_terminal_review_2026-08-01.json).

The successor-alignment hinge then produced a genuine relative improvement over its concurrent baseline, but its absolute margin remained negative and persistence still failed. See [the alignment terminal review](lewm_go2_world_model_action_alignment_successor_v1_integrity_replacement_v1_terminal_review_2026-08-01.json). Continuing from u700 to u900 worsened the primary margin; stopping that mechanism was exactly right. See [the fixed-continuation terminal review](lewm_go2_world_model_action_alignment_successor_v1_fixed_same_mechanism_continuation_v1_terminal_review_2026-08-01.json).

The correct conclusion is therefore:

> The current objective and checkpoint are not credible as a planning world model, and more updates under the same mechanism are unjustified.

It is not:

> The architecture has been cleared and lack of matched counterfactual data is now the sole explanation.

## Where the reasoning went off course

### 1. Exact state duplication was treated as an identifiability requirement

Continuous observations are almost always unique. Models can identify local action effects when nearby histories contain overlapping action support and the dynamics are sufficiently smooth or structured. The correct diagnostic is neighborhood-level action entropy and propensity overlap, not exact frame equality.

### 2. The hinge can reward action-label association without learning consequences

It asks the factual-action prediction to lie nearer the factual successor than predictions produced using wrong labels. Since it has no true wrong-action successors, it can learn "which action label belongs to this trajectory row" rather than "what each action would cause."

The observed combination of above-chance balanced accuracy and positive wrong-history separation, but negative persistence and physical ordering, is consistent with exactly that failure.

### 3. The hardest-action metric assumes action labels always imply distinct outcomes

That is invalid when commands are clipped, motions are too short to separate, actions are symmetric, or different primitives produce physically equivalent outcomes. The smoke already found safety clipping for five of nine requested primitives, despite nine distinct executed-command hashes. See [the counterfactual smoke review](lewm_go2_world_model_counterfactual_smoke_v3_terminal_review_2026-08-01.json).

Evaluation should use physical-outcome or regret-equivalence classes, not require every nominally wrong label to be worse.

### 4. Proxy optimization displaced the actual goal

Rank, variance, action decoding, hinge margins, and persistence are useful diagnostics. They became a long proxy ladder without a direct branch-truth or closed-loop planning result. The earlier June diagnosis was closer to the frontier: spatial tokens, recurrent belief/proprioception, multi-step prediction, and direct MPC were already identified as the real program. See [the June navigation plan](lewm_jepa_navigation_next_steps_2026-06-14.md).

### 5. Uncertainty is understated

The key training experiments use one seed, and the same validation scenes informed localization, treatment selection, and continuation. The bootstrap bounds are useful protocol-local evidence for stopping, but are not fresh-scene or training-seed uncertainty.

### 6. The 3 TB figure is scientifically misleading by itself

The pool contains roughly 1.81 million H6 candidates and 55.2 million frame rows, but the strongest experiment trained on only 16,000 candidates, about 0.9%. Bytes mainly measure image storage, not independent action-state coverage. Before generating or scaling anything, measure the effective conditional support already present.

## What was done well

The latest work has several real strengths:

- Scene-disjoint validation and blind/shuffled controls.
- Absolute baselines alongside relative treatment effects.
- Independent receipts, preregistration, and custody discipline.
- Explicitly recording withdrawn claims.
- Refusing to turn diagnostic B's 2.43x training-set capacity result into a generalization claim.
- Stopping the u700-to-u900 continuation when the registered metric worsened.

Those practices rescued the project from drawing a stronger false conclusion. The failure was primarily in causal interpretation and experimental prioritization, not experimental honesty.

## Recommended experimental reset

### 1. Audit the existing pool before generating bulk data

Measure, using only pre-action state/history information:

- Local k-nearest-neighbor action entropy and effective action support.
- Requested-versus-executed command discrepancies.
- Proprioceptive and ego-motion coverage.
- Dynamic-event density: meaningful displacement, contact, collision, turning, and stopping.
- Coverage by scene, not merely by frame or state count.
- Whether a state-only classifier can predict the behavior policy's action.

If actions are almost deterministic from local history, coverage is genuinely confounded. If substantial local overlap exists, the handoff's data diagnosis weakens further.

### 2. Make counterfactual generation evaluation-first

The existing smoke proves only that the plumbing can branch one state. It remains `physics_validated:false` and explicitly non-scientific.

Before a claim-bearing run, fix:

- The unimplemented calibration joiner in [`join_go2_world_model_counterfactual_pilot_v1.py`](../scripts/join_go2_world_model_counterfactual_pilot_v1.py).
- The real-data consumer's missing analyzer-provenance path in [`go2_world_model_counterfactual_pilot_v1.py`](../lewm/datasets/go2_world_model_counterfactual_pilot_v1.py).
- The smoke-only collector assumptions.
- The evaluator's inability to load current snapshots.
- Requested/executed semantics: candidate scoring may use the requested primitive plus current controller/safety state; the future executed command tape is a target/audit variable and must not be leaked into the model input.

Run the planned 160-branch calibration across all nine primitives, not only HOLD and forward. Scale to approximately 2,304 branches only if repeatability, action distinctness, physics, visual parity, and oracle diversity pass. Allocate more of those states across independent scenes.

This set should initially be a withheld causal evaluation set, not automatically new training data.

### 3. Run a controlled objective/architecture comparison on the existing pool

At minimum:

- Current conditioned baseline.
- Same architecture plus Delta-JEPA-style latent-displacement action decoding.
- A frozen dense spatial DINOv2 or V-JEPA-2.1 encoder with proprioception, AdaLN action conditioning, and a two-step predictor.

Use the same training rows, compute budget, three or more seeds, and fresh scene-disjoint evaluation. Do not bundle every new loss into one arm.

#### Assessment of the proposed self-supervised dense-token successor

Do not interrupt the active branch-truth experiment or change the thesis in
response to an unfinished result. The supervised semantic-anchor/auxiliary-head
family is already a terminal negative mechanism branch, however, so a further
BEV-label variant should not be the default successor if the current models
also fail the direct causal gates. The active experiment is an evaluator of
frozen models, not another semantic-head attempt.

The clean successor is a preregistered matched comparison, not one favored
V-JEPA arm:

- DINOv2 dense patch features versus V-JEPA 2.1 dense spatiotemporal features;
- frozen initialization versus identical robot-domain self-supervised
  adaptation for each initialization;
- one local, no-external-pretraining dense-JEPA reference retained as a
  report-only thesis-identity control; and
- one shared action-conditioned predictor, history/proprioception interface,
  optimizer budget, two-step target, seeds, planner, and evaluator across all
  cells.

The predictor should receive only pre-action RGB/history/proprioception and the
requested candidate action, with AdaLN-style action conditioning at every
predictor block. Future executed commands, semantic occupancy, depth, contact,
reward, simulator pose, and evaluator labels must not enter the strict
RGB/action training cell. An embodiment-derived variant may use measured
motion, IMU, contact, calibration, or camera height, but it must be named and
reported separately as *self-supervised visual predictive learning with
embodiment-derived geometric supervision*. That is a useful formulation, but
it is not the same scientific claim as pixels, temporal order, and actions
only.

Metric-scale wording also needs care. Discrete action IDs do not identify
metres. Commands expressed in physical units can anchor a *commanded* scale
only by assuming a calibrated actuation/dynamics model; they do not reveal the
executed displacement under slip, clipping, or contact. Measured odometry or
body displacement can identify executed metric scale, but belongs in the
separately reported embodiment-derived cell. The strict RGB/action cell should
therefore claim relative controllability, reachability, topology, temporal
distance, and branch-specific prediction unless metric scale is established by
an explicit allowed signal and ablation.

V-JEPA 2.1 initialization is also not thesis-neutral. The foundational plan
treats external pretraining as a control, so a frozen or adapted V-JEPA 2.1
winner would establish that externally pretrained video features can make this
planning stack work. It would not by itself establish that this repository
learned the useful representation. If that broader claim is desired, retain
the no-pretraining reference or explicitly amend the thesis scope before the
run.

Dense tokens alone are not the novelty: the repository already qualified a
dense single-frame spatial JEPA and obtained a negative dense temporal
predictor result. The proposed experiment is useful because it isolates
pretraining source and robot-domain adaptation while testing the complete
scientific chain: direct same-state branch truth, blind 1/2/4/8-step rollout,
oracle-headroom planning calibration, and same-sensor closed loop. If frozen
DINOv2 wins while V-JEPA 2.1 does not, the result is a planning-feature ceiling,
not evidence for a video-JEPA mechanism.

### 4. Replace proxy-only gates with the actual scientific chain

Require separate evidence for:

1. **Representation health:** no collapse; spatial and proprioceptive information retained.
2. **Action binding:** fixed-history interventions produce the correct branch-specific physical direction, including null/equivalent-action calibration.
3. **Rollout:** blind 1/2/4/8-step predictions beat persistence for planning-relevant quantities.
4. **Planning geometry:** predicted cost correlates with true physical progress/regret, with a true-future oracle demonstrating evaluator headroom.
5. **Causal deployment:** under identical sensors, the world-model planner improves navigation success over reactive, kinematic, and action-only baselines; ablating or deranging the model changes planner choices and removes the gain.

Align planner scoring with the prefix actually executed, rather than scoring a terminal horizon that is never reached before replanning.

### 5. Apply a strict progression rule

- If action-grounding objectives improve branch truth across seeds, scale training using the existing pool first.
- If diagnostics improve but planning does not, investigate reachability, latent geometry, uncertainty, and planner exploitation.
- If existing-pool methods cannot improve true branch fidelity, matched branch data becomes the leading training intervention.
- After two adequately powered non-improving mechanisms, stop JEPA-specific tuning and compare against a conventional state-space dynamics model and a Dreamer-style task-coupled baseline.

## Bottom line

The repository has not failed at JEPA navigation. It has now run a valid
scene-disjoint matched-branch **development evaluation**, though still not a
closed-loop JEPA navigation experiment. That panel found the current pooled,
one-step checkpoints are not planning-ready.

The most defensible current claim is:

> We have reproducible action-conditioned latent signal, but no demonstrated
> physical action-ranking advantage, relative-progress advantage, rollout
> utility, planning geometry, or causal navigation benefit.

The fixed branch evaluator is complete. Do not launch a bulk rerender or continue
tuning the same observational proxy. The next information-changing experiment
should train on a deliberately scoped matched-state/multiple-action branch set
and compare a dense-token JEPA candidate against conventional state-space and
Dreamer-style baselines before any closed-loop claim.

## Post-panel result update (2026-08-02)

After this review was written, the preregistered four-arm by three-seed
update-700 panel completed on the fixed matched-branch evaluator. All 12 reports
were produced. The aggregate verdict is
`USEFUL_SCENE_DISJOINT_PLANNING_EVIDENCE_NOT_ESTABLISHED`.

This resolves the review's largest open causal question:

- `masked_plain` and `full_plain` show some direct latent improvement over
  shuffled controls, so the learned representation is not simply action-blind;
- neither plain arm passes physical rank-regret, relative target-progress,
  retrieval, or falsification in any seed;
- the true-future ceiling fails evaluator sensitivity, so this is evidence that
  usefulness was not established, not proof that no useful model is possible;
- mean physical rank-regret effects are adverse (`+0.0358166` masked and
  `+0.0490916` full, where negative favors the forecast);
- delta supervision and full-grid targets both significantly worsen direct
  matched-branch error in all three seeds, so neither is a practical mechanism;
- no checkpoint is eligible for blind rollout or planning integration.

The result supports this review's warning that improved proxy prediction need
not yield plannable geometry. It also changes the data recommendation. Matched
branches are no longer merely the route to decisive evaluation; under the
frozen stop rule they are now the next information-changing **training**
intervention for this program. That is a progression decision, not proof that
observational learning is impossible or that exact duplicate states are a
general identifiability requirement.

The registered route is
`STOP_OBSERVATIONAL_MECHANISM_TUNING_AND_COLLECT_MATCHED_BRANCH_TRAINING_DATA_THEN_COMPARE_CONVENTIONAL_AND_DREAMER_BASELINES`.
A dense spatial-token, action-conditioned JEPA initialized from V-JEPA 2.1 and
controlled against frozen V-JEPA 2.1 and DINOv2 remains a defensible candidate
inside that comparison. It should not replace the conventional and Dreamer
controls or inherit a usefulness claim from pretraining.

The bound aggregate is
`.generated/dev/go2_world_model_bounded_branch_evaluation_panel_v1/panel_result.json`
(SHA-256
`5439afee451cc66ca21c991a55266aed1c430444cc0b8112a7e14155e7e57fe8`).
The independent terminal record is
`docs/lewm_go2_world_model_bounded_branch_posthoc_evaluation_v1_terminal_review_2026-08-02.json`
(16,458 bytes, SHA-256
`58cbaec33e27a65d25d0106a43f6995bc75393706c4ac99637ae3e9e0f08373e`;
status `PASS_COMPLETE_TERMINAL_DEVELOPMENT_REVIEW`).

## Matched-training mechanism update (2026-08-03)

The next controlled training experiment has now run.  It was deliberately a
train-only engineering screen, not a navigation evaluation.  The existing
matched pool supplied 128 states from 16 scenes, with all nine requested
successors per state.  Frozen V-JEPA 2.1 and DINOv2 tokens were representation
controls: they tested whether a compact action-conditioned transition could use
their spatial features.  They were not treated as policies and their feature
quality was not counted as evidence that either encoder can navigate.

The four fixed arms were a dense predictor over V-JEPA tokens, the same dense
mechanism over DINOv2 tokens, a deterministic pooled state-space model over
V-JEPA, and a compact RSSM-style model over V-JEPA.  At 800 updates every arm
had a positive intervention margin and every arm had learned substantially
from initialization, but all four failed both fixed capacity gates:

| Arm | Error / persistence (must be <= 0.80) | Branch retrieval (must be >= 0.50) | Intervention margin |
|---|---:|---:|---:|
| Dense V-JEPA 2.1 | 0.9164 | 0.2804 | 0.0287 |
| Dense DINOv2 | 0.9667 | 0.2049 | 0.0460 |
| Deterministic state-space | 0.8784 | 0.1484 | 0.0148 |
| Compact RSSM | 0.8263 | 0.1354 | 0.0106 |

The full projected 12-member comparison would have required only 0.822 GPU
hours, so compute was not the blocker.  The independently reproduced terminal
decision was `STOP_BEFORE_FRESH_MATCHED_BRANCH_COLLECTION`; see
[the four-arm terminal review](lewm_go2_matched_branch_successor_screen_v1_terminal_review_2026-08-03.json).

Dense V-JEPA was the only arm whose late curve plausibly left an optimization-
horizon ambiguity.  A separately preregistered, no-RGB diagnostic therefore
retrained that exact mechanism from the same seed, replayed the exact
update-800 witness, and imposed a conjunctive update-1,600 futility gate.  It
stopped at 1,600:

- retrieval improved from `0.28038` to `0.43056`, or from 323 to 496 correct
  action-successor matches of 1,152;
- intervention margin improved from `0.02867` to `0.04542`;
- error-to-persistence ratio improved from `0.91644` to `0.87234`; but
- the fixed fidelity midpoint was `0.85822`, so the ratio gate failed by
  `0.01412` even though retrieval and intervention passed their midpoint gates.

The correct interpretation is asymmetric progress.  The frozen-feature dense
predictor increasingly identifies which successor belongs to which requested
action on its training scenes, but its successor fidelity is not improving over
persistence quickly enough to establish the registered capacity claim.  This
is another concrete instance of the literature's warning that action
discrimination or reduced prediction loss need not produce a plannable world
model.  The independently reproduced terminal is
`COMPLETE_FUTILITY_STOP`; see
[the horizon terminal review](lewm_go2_dense_vjepa2_1_horizon_diagnostic_v1_terminal_review_2026-08-03.json).

### What this changes

1. **Do not generate the 1,024-state fresh campaign yet.**  The current
   frozen-feature mechanism did not qualify, and the 3 TB passive H6 pool does
   not supply same-state/multiple-action supervision merely by being large.
2. **Do not extend this predictor to 3,200 updates or tune its width, seed,
   optimizer, or loss weights.**  That would override a result-dependent stop.
3. **Keep the thesis and benchmark.**  The direct branch-truth evaluator,
   physical gates, scene separation, planner comparison, and memory question
   remain the right scientific chain.
4. **Change the next representation mechanism.**  The clean successor is a
   joint or self-supervised action-conditioned dense-token learner initialized
   from V-JEPA 2.1, with DINOv2 and frozen V-JEPA retained as representation
   controls.  The encoder/latent must be allowed to reshape around
   action-relevant innovation rather than asking a compact head to recover it
   from a frozen geometry.  Semantic occupancy remains an evaluation or
   qualification instrument, not the default training target.
5. **Separate candidate qualification from baseline inclusion next time.**
   Requiring both conventional controls to pass the candidate's capacity gate
   conflates “the baseline is weak” with “the comparison is not worth running.”
   Controls should remain in the matched comparison even when they are poor.
   This did not change the present terminal because the dense candidate itself
   also failed, but it should be corrected in a future preregistration.

Nothing in these screens establishes fresh-scene generalization, correct
physical action ranking, rollout composability, useful planning geometry, or
causal navigation benefit.  The project has run a real representation-and-
dynamics capacity experiment; it has still not run an experiment in which a
learned world model successfully chooses and executes navigation actions.
