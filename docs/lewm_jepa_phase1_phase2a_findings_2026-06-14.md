# JEPA Navigation Findings: What Failed, What We Learned, And What Comes Next

Date: 2026-06-14

Branch: `jepa-spatial-world-model-nav`

## Purpose

The purpose of this research is not simply to make the robot reach a goal.
Conventional mapping, depth-based obstacle avoidance, and geometric planning
could make this particular task more successful much sooner.

The research question is harder:

> Can a JEPA-style world model learn a representation of the world that is
> sufficiently spatial, predictive, and action-sensitive to choose safe,
> useful navigation actions directly?

The failed final demo matters because it showed that the surrounding
engineering could compensate for the learned model, while the learned model
itself was not reliably choosing safe paths. The new experiments therefore
remove as much controller scaffolding as possible and test the world model's
decision capability directly.

## Lay Explanation Of The Test

Imagine pausing the robot at one exact moment and asking:

- What happens if you go forward?
- What happens if you turn left and then move?
- What happens if you turn right?
- Which choice is safe and still gets you closer to the goal?

The counterfactual benchmark is that paused moment repeated many times. Every
candidate action starts from the same state, so the model cannot blame a
different starting position or a different maze.

The privileged oracle is the answer key. It is not given to the deployed
model. It tells us whether a good action actually existed and lets us measure
how often the learned model chose it.

The persistence baseline is deliberately simple: assume that the future latent
view will be the same as the current latent view. Nearby camera frames often
look similar, so a useful world model must beat this surprisingly strong
baseline. If it cannot, its imagined futures add error rather than useful
knowledge.

## What The Final Demo Actually Established

The final ego-depth demo established that the integrated system could complete
one held-out maze with substantial privileged and hand-engineered support. It
did not establish learned world-model planning.

The successful run depended on:

- a privileged scene-graph DFS route;
- simulator-rendered depth;
- ground-truth pose and mapping-time bearings;
- reactive obstacle veto and escape behavior;
- repeated heading-alignment turns;
- a pixel-space final colour servo.

It used 215 yaw primitives, 127 alignment blocks, and 64 veto escapes, and
travelled 3.21 times the direct path length. The LeWM fallback planner made no
fallback decisions in the successful artifact.

Lay interpretation: the robot reached the destination because an engineered
navigation stack repeatedly corrected it. It did not demonstrate that the JEPA
understood the maze well enough to choose a clean safe route.

## Phase 1: Direct Decision Benchmark

### Why this benchmark was required

Earlier experiments often measured proxy capabilities:

- whether pose could be decoded;
- whether a head could rank logged samples;
- whether views could be retrieved;
- whether next-latent prediction loss decreased.

Those are useful diagnostics, but none directly asks whether the model chooses
the right action. Phase 1 changed the measurement target from "does the latent
contain some useful signal?" to "does using the latent lead to a safe,
progressing decision?"

### Benchmark scale

The full benchmark contains:

- 14,890 training states from 32 scenes;
- 14,381 evaluation states from 32 disjoint scenes;
- 81 two-block action sequences from every state.

Every candidate records safety, clearance, progress, heading error, and whether
the goal remains reachable. These components remain separate so a model cannot
appear successful merely by refusing to move.

### Main result

| Selector | Newly unsafe | Ends unsafe | Mean progress | Safe positive progress |
| --- | ---: | ---: | ---: | ---: |
| action-only prior | 0.00% | 15.65% | 0.000 m | 0.00% |
| random expected | 25.34% | 34.21% | 0.053 m | 35.65% |
| logged then hold | 20.20% | 34.31% | 0.059 m | 36.74% |
| current pooled LeWM rollout | 14.26% | 24.10% | 0.027 m | 23.13% |
| transparent oracle | 0.00% | 2.69% | 0.138 m | 85.21% |

The current pooled LeWM is less unsafe than random, but mostly because it is
more conservative and makes less progress. It achieves safe positive progress
less often than random or replaying the logged action. It exactly matches the
oracle sequence only 2.50% of the time.

Lay interpretation: the model has learned something closer to "be cautious"
than "this opening is safe and leads toward the goal." A driver who rarely
accelerates may crash less often than a random driver, but that is not the same
as knowing how to drive.

### What the oracle proves

The oracle achieves safe positive progress on 85.21% of goal-conditioned
states. This is important because it shows that the action set and horizon
usually contain a useful answer. The learned model is not failing merely
because every available action is bad.

### Physics calibration

The exhaustive benchmark uses a cheap kinematic approximation. A bounded
Genesis replay of 32 balanced candidates found:

- 0% falls;
- 0.094 m mean endpoint error;
- 0.093 rad mean yaw error;
- 62.5% newly-unsafe label agreement;
- 75.0% ends-unsafe label agreement.

The kinematic proxy is useful for generating and screening many branches, but
it is too inaccurate to be the sole physical-safety label.

Lay interpretation: the proxy is a useful pencil sketch of where the robot
might go, but it is not a trustworthy crash test.

## Why The Pooled Base World Model Is Not Appropriate As-Is

The existing LeWM compresses each image into one CLS vector and predicts future
CLS vectors. This can support coarse appearance recognition, but navigation
requires facts whose location matters:

- the wall is on the left;
- the centre is open;
- the safe gap is behind a nearby obstacle;
- a small rotation changes clearance;
- two visually similar corridors lead to different outcomes.

Lay interpretation: global pooling is like describing a room with one sentence.
"A corridor with grey walls" may be adequate for recognizing the place, but it
does not say where the doorway is or which side contains the obstacle.

This is not evidence that JEPA is unsuitable. It is evidence that this
particular globally pooled JEPA state is insufficient for spatial action
selection.

## Is SIGReg Working Against The Task?

SIGReg is doing its intended job: it discourages a trivial collapsed
representation by encouraging projected CLS embeddings to resemble an
isotropic Gaussian.

However, navigation geometry is not naturally isotropic. It contains:

- cyclic heading;
- local metric neighbourhoods;
- topological branches;
- sharp collision boundaries;
- different invariance needs for place recognition and local control.

Previous scaled ablations showed that merely lowering the SIGReg coefficient
did not recover navigation geometry. Lower SIGReg degraded retrieval,
dispersion, and action sensitivity while projected geometry remained poor.

The correct conclusion is therefore not "SIGReg is bad." The problem is that
one pooled branch receives one uniform regularisation rule while being asked to
serve conflicting roles. The next model should apply anti-collapse according
to branch semantics and select it using decision outcomes, not latent
dispersion alone.

Lay interpretation: anti-collapse is like requiring a class to use the whole
range of exam scores. That prevents everyone receiving the same score, but it
does not guarantee that the exam measures navigation ability.

## Why Learned Heads And Planner Heuristics Did Not Solve It

The learned heads could often improve their own proxy objective without
improving deployed decisions. A small head on a frozen latent cannot reliably
recover spatial information that the encoder discarded or never organized.

Additional planner heuristics then compensated for model errors using spins,
vetoes, recovery rules, privileged bearings, and local geometry. Those
heuristics can improve task success, but they stop answering the PhD research
question because the world model is no longer the source of competent action
selection.

The right use of heads is as diagnostic probes or as jointly trained,
semantically justified branches that change the learned representation. The
wrong use is repeatedly adding frozen-latent heads until one fits a proxy.

## Phase 2A: Frozen Spatial-Token Diagnostic

### Hypothesis

The first spatial experiment asked whether the existing encoder already
contained a usable spatial state in its patch tokens, even though the deployed
model pooled them into CLS.

This was deliberately conservative. It retained the old encoder and trained
only an action-conditioned predictor over the ordered 16 by 16 patch-token
grid. The default six-layer predictor is capacity matched to the pooled
predictor within 0.44%.

If this worked, it would show that pooling was the main loss. If it failed, it
would show that simply exposing old internal patch activations is insufficient.

### Future-observation data finding

A scaled bounded render used 64 states and nine candidate sequences per state
in one training scene and one disjoint evaluation scene. Each sequence has two
future observation slots.

- training: 416 of 1,152 planned future frames invalid;
- evaluation: 311 of 1,152 planned future frames invalid.

Invalidity is strongly outcome-dependent:

| Split | Safe sequence has any invalid frame | Unsafe sequence has any invalid frame |
| --- | ---: | ---: |
| train | 31.2% | 56.5% |
| evaluation | 21.1% | 52.9% |

Silently dropping invalid sequences would therefore remove unsafe examples much
more often and create an artificially easy safety benchmark.

Invalid render observations are not themselves collision labels. Many
kinematically safe branches are invalid because the replay camera is near
forward geometry, and the kinematic safety proxy is already known to be noisy.
The corrected dataset contract now preserves every candidate, marks missing or
renderer-invalid observations explicitly, and limits token loss to valid
observations.

Lay interpretation: there is no honest normal photograph of "after the camera
has moved into the wall." Throwing those cases away would train only on the
easy journeys. Calling every missing photograph a crash would also be wrong.
The model eventually needs to predict both valid future state and terminal or
contact events, with those events calibrated by physics.

### Predictor implementation control

The spatial predictor initially inherited causal attention from the temporal
LeWM block. That mask is correct for a time sequence but incorrect for a patch
grid because it prevents early flattened patches from attending to later
patches. The spatial predictor was corrected to use bidirectional patch
attention while preserving causal attention for the temporal pooled model.

The corrected run produced materially the same negative result, so the causal
mask was a real implementation confound but not the explanation for failure.

### Bounded corrected result

The corrected scene-disjoint diagnostic used:

- 338 complete valid training sequences;
- 399 complete valid evaluation sequences;
- a one-layer spatial predictor as a bounded learnability test;
- the frozen existing encoder;
- no privileged consequence labels in training.

At epoch 5:

- training token MSE: `0.210`;
- held-out teacher-forced token MSE: `0.755`;
- held-out free-running token MSE: `0.955`;
- held-out persistence token MSE: `0.347`;
- free-running/persistence error ratio: `2.76x`.

It fails immediately:

| Horizon | Spatial predictor MSE | Persistence MSE |
| --- | ---: | ---: |
| one block | 0.782 | 0.324 |
| two blocks, free-running | 1.129 | 0.370 |

The held-out action selection diagnostic is conditional on complete valid
future observations and is not a safety result. Within that filtered subset it
achieves:

- 21.95% safe positive progress;
- 7.32% newly unsafe selections;
- 21.95% ends-unsafe selections;
- 0% oracle-sequence match;
- +0.005 m mean target progress.

### Interpretation

The predictor fits the training scene but cannot predict held-out patch-token
change better than assuming the latent remains fixed. Because it loses at the
first step, recursive rollout drift is not the primary cause.

The most important architectural fact is that the original LeWM training loss
only predicts and regularises the CLS representation:

- `VisionEncoder.forward()` returns only CLS;
- the LeWM predictor receives only CLS sequences;
- prediction MSE is applied to projected CLS;
- SIGReg is applied to projected CLS;
- patch tokens receive no direct spatial correspondence or future-prediction
  objective.

Patch tokens therefore exist, but the training contract never required them to
be stable spatial state. Their grid position reflects image patches, yet their
features are free to reorganize in ways that are useful to CLS aggregation but
poor for action-conditioned prediction.

Lay interpretation: opening the old model and finding a grid of internal notes
does not mean those notes form a map. They were written to help produce one
summary sentence, not to remain consistent as the robot moves.

## What Is Established And What Is Not

Established:

- the current pooled LeWM is not a competitive safe-action decision
  representation;
- useful actions usually exist in the candidate set;
- kinematic safety labels require physical calibration;
- invalid future observations are outcome-dependent and cannot be silently
  dropped;
- exposing frozen old patch tokens does not beat persistence, even at one
  step;
- increasing frozen-predictor capacity is not justified yet.

Not established:

- that a JEPA trained end-to-end on spatial tokens will fail;
- that SIGReg should be removed;
- that patch-token prediction is the best spatial representation;
- that renderer invalidity corresponds to physical collision;
- that the one-scene bounded Phase 2A result estimates final generalization.

## Research Decision

Do not scale the frozen spatial predictor and do not add another learned head
to the old encoder.

Proceed to an end-to-end spatial JEPA experiment whose objective explicitly
creates predictable spatial state. The experiment must remain capacity matched,
scene disjoint, counterfactual, and directly gated against persistence and the
pooled LeWM.

## Next Falsifiable Experiment: Phase 2B

### Hypothesis

An encoder trained end-to-end to produce action-predictable spatial tokens will
beat persistence and pooled LeWM on held-out counterfactual future prediction
and action ranking.

### Controlled variants

Train matched-capacity variants:

1. pooled CLS LeWM control;
2. spatial-token JEPA with a pooled appearance token and anti-collapse applied
   only to the appearance branch;
3. the same spatial model with a token-local variance floor;
4. optionally, uniform token SIGReg as a negative or comparison control.

Do not add recurrent belief, long-range topology, decoded occupancy, or a
complex planner during this experiment. The purpose is to isolate whether the
representation can become spatial and action-predictable.

### Data correction

Generate broader scene-disjoint matched branches with:

- balanced safe-progress, unsafe, recovery, rotation, and hold outcomes;
- valid future observations where meaningful;
- explicit missing/invalid observation events;
- bounded physical replay for terminal/contact calibration.

Token prediction loss is applied only where a valid future observation exists.
Event prediction must remain a separate target and must distinguish renderer
validity, kinematic unsafe labels, and physics-validated contact.

### Promotion gates

Before adding recurrence, the end-to-end spatial JEPA must:

- beat persistence at one block on disjoint scenes;
- beat persistence in free-running two-block prediction;
- outperform the pooled LeWM on complete counterfactual action ranking;
- improve safe positive progress without increasing newly unsafe selection;
- retain explicit reporting of excluded or invalid observation coverage;
- show that gains survive a physics-calibrated safety subset.

If it cannot beat persistence at one block after controlled end-to-end
training, redesign the spatial state or target before increasing model size or
horizon.

## Longer-Term Plan

Only after Phase 2B passes:

1. add recurrent belief for hidden and out-of-view state;
2. train longer free-running counterfactual futures;
3. factorize appearance, affordance, dynamics, and belief roles;
4. compare branch-specific anti-collapse strategies;
5. make latent MPC the primary controller;
6. restore long-range image-goal navigation without privileged route replay.

This sequence preserves the project's JEPA research purpose while preventing
another long cycle of controller heuristics compensating for an unsuitable
representation.

## Implementation Status At This Decision

The repository now contains the first Phase 2B implementation unit:

- an end-to-end `SpatialLeWorldModel`;
- a default complete-model capacity match within `+0.27%` of pooled LeWM;
- patch-token action-conditioned prediction loss;
- separate appearance CLS SIGReg;
- a spatial token-position-local variance-floor loss;
- free-running projected spatial rollout;
- outcome-stratified bounded counterfactual render-plan selection;
- focused tests for gradients, shapes, anti-collapse behavior, attention
  semantics, invalid-future retention, and balanced candidate selection.

No Phase 2B training result is claimed yet. The implementation establishes the
controlled experiment required to obtain one.
