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

The repository has not failed at JEPA navigation; it has not yet run a valid JEPA navigation experiment. It has run increasingly rigorous component diagnostics and found that the current pooled, one-step objective is not planning-ready.

The most defensible current claim is:

> We have factual action association, but no demonstrated counterfactual action fidelity, rollout utility, planning geometry, or causal navigation benefit.

Do not launch a bulk rendering job yet. Finish the small branch-truth evaluator, audit action support in the existing pool, and compare one action-grounding objective and one dense spatial baseline. That is the shortest path to determining whether the limitation is data coverage, objective, representation, or planning, and it avoids another round of optimizing a proxy that never reaches the robot.
