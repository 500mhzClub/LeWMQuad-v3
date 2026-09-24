# Separate frozen representation quality from latent-transition fitting

**All three fits and branch evaluations are complete.** No observation encoder
or target changed. Common predictor initialization matched exactly across arms;
all cached-forward and checkpoint-reload comparisons were exact. Peak RSS was
10.07–10.09 GB per worker; each completed its 1,200 updates in about 90 seconds
including input loading and feature encoding. All jobs exited zero.

| Frozen representation | Weighted training latent MSE before | After | Transfer 800-ms latent MSE before | After | Action retrieval before → after |
| --- | ---: | ---: | ---: | ---: | --- |
| JEPA | 0.045388 | 0.002355 | 0.021411 | 0.002098 | 6/18 → 11/18 |
| Supervised | 18.715638 | 0.063272 | 54.529404 | 0.052762 | 6/18 → 7/18 |
| Untrained | 3.711873 | 0.000981 | 7.801778 | 0.000649 | 6/18 → 7/18 |

Raw errors across rows use different target representations; compare within
rows and against each representation's baselines. Supervised/untrained original
transitions were never optimized to predict EMA embeddings, so their original
errors do not establish JEPA superiority. All encoded future and substituted
scene targets match the prior assay exactly after these predictor refits.

JEPA transfer error falls about 90.2%. It now beats its training horizon mean
(0.002911), but remains worse than the preceding-action/horizon training mean
(0.000453). Its centered action-effect MSE is 0.884 times the action-independent
baseline, versus 1.647 before refitting. Supervised/untrained ratios after the
matched refit are 1.036/1.082. These are small exposed-development diagnostics,
not a statistically established JEPA advantage. JEPA scene retrieval is 11/18
versus persistence's 10/18; static appearance and tiny populations remain limits.

The old motion readouts degrade, as anticipated: JEPA's unadapted 800-ms XY
RMSE is 70.29 mm. This readout no longer matches its changed predictor features.
Do not promote it or interpret the unadapted head as the final downstream
comparison. The separate matched readout adaptation below uses training only.

Result: `docs/go2_frozen_representation_dynamics_result_2026-09-17.json`.
Per-arm final weights, update losses, exact input identities and fitting results
are retained. `branch_evaluation/` contains complete curves, post-hoc baseline
decomposition and exact representation-equality evidence.

The completed action-branch assay found that current JEPA prediction beats
persistence but loses to training-only target means and does not reliably
identify action branches or visual scenes. This experiment addresses that
model-learning failure while keeping perception and native navigation unchanged.

Three fixed fits use the original JEPA, matched supervised, and untrained
observation representations. Freeze each online sensory encoder and EMA target
encoder, and every motion/contact/reference component. Reset history GRU,
history normalization and action-conditioned transition to the exact common
untrained initialization in all three arms. Fit only those three modules on
latent prediction loss: mean squared error per available future, averaged
within each context and then across contexts with targets. Do not update EMA
targets or include supervised motion/contact losses. This is a matched latent
predictor refit on three different frozen representations, not three new
end-to-end representations or a standalone JEPA superiority test.

Use the unchanged 4,694 training contexts, 7,200 scheduled draws, 1,200 updates,
six-context batches and original schedule. AdamW learning rate is fixed at
0.001, zero weight decay, gradient clipping at norm 1. No schedule/hyperparameter
search, intermediate checkpoint selection, transfer fitting or navigation-data
training occurs. All three fits use every scheduled draw. Contexts without
future observations stay accounted for and do not contribute a latent label.

Frozen features are computed transiently from the same original policy inputs
and actual available future observations. Cached-forward equality is checked
against the original model on the first input batch. Target masks remain
explicit. Once features are encoded, release the RGB tensor cache. Preserve
small predictor weights, update losses, input identities and final results.
Verify frozen state identity and reloaded predictor equality after training.

Evaluate all three final models only after all fits finish, on the unchanged
36 frame-13 action-branch contexts (18 train, 18 exposed development transfer).
Report the same complete horizon curves, with 800 ms primary: persistence,
wrong actions, branch retrieval, scene retrieval and RGB intervention. Retain
training-mean baselines and the centered action-effect diagnostic. All encoded
future and scene-control targets must exactly match the preceding assay, so
within-representation before/after errors use a common target space.

The old motion readout is stale after the predictor changes. Its computed
scores are only an unadapted-head diagnostic; no navigation promotion follows.
A useful latent improvement would need a training-only readout fit and then
a prospective matched navigation test. Stronger training-only means and
action/scene specificity remain necessary comparisons before interpreting a
lower latent loss as useful world modeling.

Plan: `docs/go2_frozen_representation_dynamics_plan_2026-09-17.json`.
Runner: `scripts/train_go2_frozen_representation_dynamics_development.py`.
Output: `go2_frozen_representation_dynamics_v1_attempt_001` on the artifact drive.

Before launch: Ryzen 9950X3D, 72 GiB RAM available, CPU 96% idle, both GPUs idle,
4.4 GiB artifact-volume space. Three independent CPU fits use cores 8/0/1 with
one numerical-library thread each. Expected peak per worker is around 10 GiB
from the identical earlier data-loading path; monitor actual use. No native
simulation runs concurrently, and no new depth or reusable image-tensor archive
is generated. This fits the available RAM/storage without a new collection.
