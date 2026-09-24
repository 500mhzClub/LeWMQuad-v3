# RGB/body direct and JEPA reference preparation

Historical preparation record. The fixed comparison has now completed; see
the [audited result](go2_rgb_body_learning_comparison_development_v1_result_2026-09-05.md).
The original preparation notes below describe its pre-fitting status.

Implemented, with synthetic tests; **not yet fitted or compared on the maze
corpus**. The small reference uses a spatial CNN, ordered body/control GRUs, a
shared observation latent, a direct outcome head, and an action-conditioned
latent transition. Its future targets use a stop-gradient EMA encoder. This
adapts the predictive-embedding idea, not the image masking experiment of
[I-JEPA](https://arxiv.org/abs/2301.08243). Variance/covariance penalties are
available following [VICReg](https://arxiv.org/abs/2105.04906); they are not a
guarantee of noncollapse or useful navigation.

The strict observation adapter admits only RGB, chronological ideal body history
and known past applied commands. It preserves spatial image layout and body
sample order, with explicit validity and age. Known candidate plans are separate
inputs. Direct prediction encodes only the known action prefix and never recurses
a predicted world state. A dedicated direct-inference method does not execute
the transition model. Latent rollout predicts every 0.5 s autoregressively.

Both outcome interfaces return relative x/y, sin/cos heading change and a contact
logit. Censored motion targets are indexed out before arithmetic; no terminal
state is imputed as an unobserved future. The recorded displacement is projected
with the branch-start body rotation, while heading change is wrapped world-yaw
difference; this is a planar heading target, not a full relative SO(3) rotation.

Tests cover input rejection, chronology, future-action causality, EMA/stop-gradient
behavior, censored NaNs and collapsed embeddings. They include a synthetic CPU
optimizer step solely to test gradient flow. They do not establish prediction
quality, absence of collapse after fitting, executed decision value or transfer.

Before fitting, freeze the actual training procedure, roles, seeds, optimization
budget, loss weights and checkpoints after the corpus passes raw audit. Compare
nonpredictive versus predictive training with the same observation encoder and
direct head, then direct inference versus rollout from the predictive model.
Report active parameter counts and inference cost; these heads are comparable
small references, not falsely claimed to have identical active parameter counts.
Ensure both training conditions see the same available observation population,
not predictive training alone seeing additional future images. Validation groups
are layouts, not frames or alternative actions. No final test data are involved.

Retain kinematics/action-only, persistence, sensor/action-shuffle controls and
continuous executed-outcome metrics. Conditional action regret must distinguish
any oracle local intent from a target supplied by online exploration/memory.
These preparations support the scientific comparison; they do not complete it.
