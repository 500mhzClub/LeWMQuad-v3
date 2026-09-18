# JEPA world-model literature review and research direction

Reviewed 2026-09-17 following the user's request to reconsider the scientific
direction. This is a focused primary-source review, not an exhaustive novelty
search. No experiment or collection is launched by this note.

## Correction after recovering prior local experiments

The initial recommendation to establish new LeWM/DINO baselines was premature.
The user identified substantial prior V-JEPA 2.1 experiments. Their existing
result JSONs and reports were located and read on 2026-09-17. These supersede the
initial next-step ordering below; the recent small CNN/GRU line is not the full
experimental history.

- The frozen dense screen used 4,262 training frames from 72 scenes and 495
  selection frames from eight disjoint scenes. Occupied IoU was **0.510294 for
  V-JEPA 2.1 ViT-L**, **0.470854 for DINOv2 ViT-L**, and **0.372419 for the
  project ViT**. These values were read directly from
  `.generated/dev/DEVELOPMENT_ONLY_frozen_dense_representation_screen_v1/result.json`.
  All three named encoder checkpoints and feature caches still exist. V-JEPA
  and DINO had similar parameter counts but different resolution/token density;
  the comparison does not isolate pretraining objective. See
  `lewm_go2_frozen_dense_representation_screen_result_2026-08-06.md`.
- Action-free spatial JEPA and subsequent temporal joint training already ran.
  The July 31 handoff reports degraded spatial-control scores after joint
  temporal optimization, without attributing the loss to encoder versus
  predictor. The August 1 three-arm experiment then held the project encoder
  fixed while comparing conditioned, blind and shuffled predictors. See
  `lewm_go2_world_model_session_handoff_2026-07-31.md` and
  `lewm_go2_world_model_existing_pool_three_arm_v1_integrity_replacement_v3_terminal_handoff_2026-08-01.md`.
- The later V-JEPA 2.1 frozen-versus-moving comparison requires a different
  interpretation: fresh-probe true-token IoU rose **0.5010 to 0.5082**, while
  correct-versus-shuffled action margin fell **0.0586 to 0.0488**. Thus this run
  reduced action discrimination, not measured recoverable current geometry.
  Early future-token predictions lost to persistence. See
  `lewm_go2_v03_temporal_action_jepa_result_2026-08-06.md`, including its later
  alignment correction; do not stop at its initial interpretation.
- Longer training subsequently produced future occupied IoU **0.3560** for the
  selected two-step model and **0.3606** for the converged one-step control,
  versus matched persistence **0.3128**. The selection record at
  `/home/andrewknowles/.cache/lewm_go2_temporal_v03/two_step/selected_frozen_models.json`
  still exists and was read. The horizon result in the same cache's
  `horizons/evaluation/result.json` confirms positive action margins and
  prediction-over-persistence through four steps, with differing fidelity and
  discrimination rankings. These are development results, not navigation proof.
- The eight-seed factorial's surviving `factorial_v1/final_analysis.json`
  confirms a rollout H2 cosine benefit **0.008077**, 95% interval
  **[0.006428, 0.009725]**. Its occupancy co-outcome was defective and remains
  unusable. See `lewm_go2_proprio_factorial_final_report_2026-08-10.md`.

Next recover the existing selected dense predictor and its downstream decisions,
including the August counterfactual evaluation, before proposing new training.
Use frozen V-JEPA 2.1 as the evidence-backed starting candidate, retaining DINO
as the existing comparator. Establish applicability to the present native camera,
command timing and navigation decision interface rather than assume old results
transfer unchanged. Do not reinstate the superseded BEV programme merely because
its geometry probe is useful evidence. A new pooled LeWM baseline is deferred.
The untested projected-LeWM draft created during this turn was removed; no model
training or simulation was launched.

## Relevant published recipes

| Source | Relevant technique | Implication for our next comparison |
| --- | --- | --- |
| [LeWorldModel](https://arxiv.org/html/2603.19312v1), [official code](https://github.com/lucas-maes/le-wm) | Approximately 15M parameters; ViT encoder and action-conditioned transformer; prediction plus SIGReg; no EMA or stop-gradient. Uses a compact global embedding, not spatial patch prediction. | Establish a faithful end-to-end reference before interpreting custom-model failures. Projectors, normalization, temporal sampling and training budget are part of the recipe. |
| [DINO-WM, ICML 2025](https://proceedings.mlr.press/v267/zhou25t.html) | Frozen DINOv2 spatial features with learned action-conditioned dynamics and latent goal planning. | A frozen visual-feature baseline can separate representation learning failures from dynamics-learning failures. Its spatial representation is an alternative to our 32-dimensional fused state. |
| [What Drives Success in Physical Planning with Joint-Embedding Predictive World Models?](https://arxiv.org/html/2512.24497v3), [official code](https://github.com/facebookresearch/jepa-wms) | Systematic study of conditioning, history, proprioception, rollout training and planning. Its navigation recipe uses DINOv2-S, AdaLN conditioning and two-step rollout training. | Use established settings as a starting point rather than varying components without a reference. Findings are task-dependent, not universal guarantees. |
| [V-JEPA 2](https://arxiv.org/html/2506.09985v1) | Action-conditioned adaptation with frozen visual features, action/proprioception tokens and teacher-forcing plus rollout losses. | Transfer the modality separation and rollout-training ideas; reproducing the largest model is not necessary to test them. |
| [PLDM](https://arxiv.org/html/2502.14819v1) | JEPA latent dynamics, inverse dynamics and planning from reward-free offline data, including unseen-layout navigation. | Neither action-conditioned JEPA nor maze generalization alone constitutes our novelty. Its navigation observations/dynamics differ from our egocentric Go2 setting. |
| [Delta-JEPA](https://arxiv.org/abs/2606.31232) and [Sensorimotor World Models](https://arxiv.org/abs/2606.20104), June 2026 preprints | Recover executed actions from latent differences or successive representations to encourage action-relevant features. | A motivated later hypothesis for our weak action sensitivity. Action recovery must use visual representations, not simply read commands embedded in the target. Abstract-level review of these two papers; inspect methods before implementation. |

## What the current evidence actually establishes

The recent CNN/GRU, EMA-target, 32-dimensional models and frozen-predictor
variants are custom prototypes. They are not faithful reproductions of the
LeWM or DINO-WM recipes above. Their negative results diagnose those prototypes;
they do not establish that JEPA cannot learn useful Go2 dynamics. This statement
concerns the recent experiments, not every historical model in this repository.

Our current navigation system also uses predicted physical motion with explicit
geometry/routing. That differs from optimizing a goal-image distance directly
in latent space. Navigation success therefore cannot by itself validate the
published latent-planning mechanism or isolate a JEPA representation benefit.

The 90/7,200 matched-departure training-draw count motivates a coverage question,
but does not prove inadequate data or establish a requirement for identical-state
branch collection. The reviewed methods can learn from ordinary exploratory
trajectories. Sustained commands already exist in our training data. Examine
temporal alignment, observable action effects and coverage before concluding
that more matched branches are necessary.

## Potential contribution, not an established novelty claim

The useful scientific question is: when does action-conditioned JEPA prediction
improve navigation under partial observability, beyond reactive control and
explicit memory? Our setting combines an egocentric camera, body/gait-dependent
responses to velocity commands, and exploration followed by return. The robot
platform and the combination alone are not sufficient novelty evidence.

Possible contributions include a controlled account of memory versus prediction,
an action-sensitive representation adapted to temporally extended commands, or
better transfer/data efficiency under limited observations. Each needs positive
evidence against strong baselines. This review does not establish first-of-kind
status. Related visual-navigation work also exists, including
[PiJEPA, CVPR 2026 workshop](https://openaccess.thecvf.com/content/CVPR2026W/WDFM-EAI/html/Chahe_Policy-Guided_World_Model_Planning_for_Language-Conditioned_Visual_Navigation_CVPRW_2026_paper.html).

## Initial proposed order (superseded by recovered evidence above)

1. Reconcile the recent implementation against the official LeWM recipe and
   select a faithful baseline, documenting only unavoidable Go2 adaptations.
2. Establish a frozen DINO-WM/JEPA-WMs feature baseline to isolate dynamics
   learning. Account for external pretraining when comparing data efficiency.
3. Use existing development data first where suitable. Make command/frame timing
   and meaningful temporal displacement explicit. Collect more only to address
   a demonstrated coverage gap. The proposed four-tick branch collection is
   deferred, not cancelled or already authorized as the next automatic job.
4. Evaluate persistence, no-action and shuffled-action controls in a common
   representation space; use common downstream targets across different spaces.
   Test decision quality and prospective planning benefit when predictions warrant
   it. Overlapping replay windows are not independent navigation replications.
5. Test short rollout training or inverse dynamics as a focused follow-up,
   rather than combine several new mechanisms before establishing a baseline.

Keep sensor robustness, sim-to-real work, expanded candidate counts and non-maze
experiments deferred. Existing results remain valid exploratory evidence. No
JEPA superiority or deployment readiness is established.
