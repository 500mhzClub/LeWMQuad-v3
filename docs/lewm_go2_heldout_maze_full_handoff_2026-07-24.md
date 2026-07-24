# Go2 held-out maze full handoff

Date: 2026-07-24 10:03 BST

Repository snapshot:

- branch: `jepa-spatial-world-model-nav`
- HEAD before this handoff: `6e83dc6308e742becf86dbe465cf3650fa85f21a`
- persistent objective: reach one authorized opaque held-out maze test
- objective status: **blocked at the learned perception architecture gate**
- Camera checkpoint qualified for Shared/G2 promotion: **none**
- Shared-JEPA checkpoint eligible for G2: **none**
- G2: **not attempted by the current candidate**
- learned navigation runtime: **source-reviewed, not runtime-qualified**
- clean-HEAD runner/G2 dependency closure: **not achieved**
- authoritative V4 sealed set: **30 opaque scenes, frozen, unopened, and
  unmaterialized**
- active training, test, benchmark, render, navigation, or diagnostic process:
  **none observed**

## Plain-language state

The repository is not ready for held-out maze benchmarking.

No raw-data corruption or refinement problem is evidenced. The exact raw
supervision build passed its independent audit, the development roles are
fixed, and the physical gate is provably attainable by an oracle. Whether the
training distribution is adequate for a different architecture remains open.
The source-only navigation stack is also far enough along that it should be
reused rather than redesigned. A small source-closure task remains because its
runner currently imports untracked G2/publication policy files, so a clean
checkout is not yet executable.

The blocker is learned camera perception on disjoint development data.

An exact-train-only selected `N=320` camera fit passed all 26 of its fit checks
and was migrated into the Shared-V5 model. That success did not transfer to
the scene-disjoint checkpoint-selection population. Joint Camera+JEPA training
completed 8,000 updates but produced no checkpoint that passed any of the nine
complete physical scopes. A diagnostic then showed that the Camera gradient
was roughly eight orders of magnitude larger than the JEPA gradient under one
global clip, so the Camera and JEPA stages were separated.

The separated Camera stage was not JEPA training. It updated only the shared
visual encoder and Camera evidence head while freezing the BEV decoder,
predictor, target encoder, target BEV decoder, and occupancy head. The final
Camera V6 run completed all 8,000 updates and improved many individual
metrics, but still closed `0/9` complete scopes. Its direct ray and ground
signals were strong in aggregate, while its raster and rough-motion behavior
remained inadequate.

One zero-training diagnostic tested the leading cheap explanation: that a
fixed soft evidence-to-raster conversion was discarding otherwise good
information. The fixed hard MAP/Boolean conversion made aggregate raster
accuracy and occupied recall worse. That hypothesis is now rejected.

The correct next action is not another V7, longer schedule, seed, learning-rate
change, threshold change, data cleanup, or full training run. It is one
explicit decision about a materially different perception architecture,
followed by a cheap preregistered falsification before any long run.

## What the many version numbers actually mean

There were many versioned files and receipts, but they were not all distinct
scientific experiments. A large fraction were source-review, immutable-output,
path-handling, visibility, transaction, or failure-reporting corrections.

The material scientific hypotheses tried since the previous handoff were:

1. add gate-aligned raster NLL to the small Camera fit;
2. migrate that fit into Shared V5 and jointly train Camera plus JEPA;
3. separate supervised Camera adaptation from later JEPA training because the
   joint gradient scale and clip were invalid;
4. test a real Camera encoder learning-rate-scale ladder:
   `0.01` in V1, `0.1` in V2, and `1.0` in V3;
5. replace the within-bin smooth-L1 depth term with a tail-depth p95/CVaR
   objective;
6. test, without training, whether hard evidence rasterization recovers a
   large cross-scope signal.

Only items 1 through 6 should be treated as science changes. The Raw V13 data
audit passed before these runs. Continued data polishing is not supported by
the current evidence.

## Goal and definition of done

The end goal remains one frozen evaluation on the existing authoritative V4
sealed role. It is not merely a low training loss, a good Camera checkpoint, a
successful development scene, or a source-reviewed runner.

Readiness requires, in order:

1. a Camera/perception candidate that passes the unchanged disjoint physical
   development gate;
2. a trained Shared-JEPA candidate with valid selection and calibration;
3. one untouched G2 perception qualification for the selected Shared
   checkpoint;
4. exact binding of the G2-qualified checkpoint, calibrations, target head, and
   existing reviewed development runner;
5. passing G3 fast coverage, mandatory G4 learned frontier value, isolated G5
   target conversion, G6 full development navigation, and G7 robustness;
6. a complete code/model/calibration/threshold/environment freeze;
7. preservation and independent custody of the frozen V4 sealed role; and
8. one held-out execution whose result is published whether it passes or
   fails, with no tuning, retry, repair, or reselection from that result.

The matched no-JEPA arm is not a committed promotion gate. Run it only if the
final scientific claim explicitly retains a causal claim that JEPA, rather than
the trained stack as a whole, improves performance. It should not double the
default route to G2.

The governing committed contract is
`docs/lewm_go2_generalization_execution_contract_2026-07-09.md` at repository
HEAD, not its pre-existing dirty worktree edit and not the untracked 2026-07-14
goal. Its active downstream gates are:

- **G0:** passed for V4 benchmark integrity.
- **G1:** passed on V4 development: `96/96` canonical claims, `24/24`
  all-four scenes, and zero stalls or collision segments.
- **G2:** planner-admitted free precision `>= 0.99`, obstacle recall within
  `2 m >= 0.95`, useful traversable-space recall `>= 0.90`, and the unchanged
  oracle-map collision gate. The historical dataset-v2 candidate failed
  offline; the current Shared-V5 Camera line has not reached untouched G2.
- **G3:** 600 ticks on the fixed fast panel, median normalized coverage at
  least `2x` the `2.6316%` baseline (the ledger's numeric target is
  `> 5.263%`), lower scene-clustered confidence bound above no improvement,
  and no safety regression. The baseline was `1.7505%` mean normalized AUC,
  `4/32` strict claims, and `0/8` solves.
- **G4:** on scene-disjoint development evidence, not G8 sealed payload,
  frontier ranking beats distance-only and random reachable-frontier baselines
  on oracle future coverage/discovery labels; the learned head beats
  deterministic information gain on 600-tick normalized coverage; and DAgger
  closes the model-visited-state gap without scene leakage.
- **G5:** sight-to-valid-claim conversion `>= 90%`, false physical accepts
  `< 1%`, the learned observation/belief stack replaces fixed RGB masks and
  privileged target geometry, and the target stack passes in isolation under
  oracle coverage. The historical baseline was only `43-46%`.
- **G6:** at least 24 development scenes at 2,400 ticks, physically verified
  claim rate `>= 90%`, at least `75%` of scenes (`18/24`) finishing four of
  four, and no family collapse. The historical full18 baseline was `0/18`
  solves and `8/72` canonical strict claim-plus-LOS events. The intermediate
  distance-only recount was `13/72` and is not the promoted score.
- **G7:** calibrated odometry noise, deployment-equivalent locomotion,
  action-source tracing, simulator-only geometry guards disabled, and physical
  smoke.
- **G8:** freeze and execute the sealed role once.

The V4 benchmark is already committed:

- roles: `138` train, `24` development, `30` sealed;
- candidate-plan seed / SHA-256: `2026070923` /
  `0d39c62bb6c70b5143f341f82d45ada5c4ef0f4733878ac07f3d6518f64cd4b1`;
- split seed: `2026070924`;
- development-manifest file SHA-256:
  `563f240a023309af42a05a9a8f29008f02a0629dee9f77f03568f779d1166d41`;
- role-commitment content / file SHA-256:
  `12f203f6dc03dd2f0ed76067075baaac026d9b2843d9471962a9364514bf0cc7`
  /
  `82c4a9a382452031febb712faaa90bb52c8fc5e2fab2a33d8a7ea2d447413b75`;
- sealed-screening file SHA-256:
  `db118ca435877b06dfe2666b681e3313d0c5c120ed2a316853413469a5a0f103`;
- sealed commitment:
  `d2dcbcc5444f0046be41311c0127943b63c2485c39b69844ed662f79aa13fef7`;
- creation-report file SHA-256:
  `f8cfc19fe97eb1f46a3fa514b13d2b69ac50f34ebade1afbe78dd4b08222702d`;
  and
- physical eligibility: `192/192` disc and `192/192` exact polygon, with zero
  exclusions.

Only V4 train and development were materialized. Its sealed manifest remains
unopened, unevaluated, and unmaterialized. `phase4_full18` is development
evidence, and the earlier V3 sealed role was invalidated unopened because its
geometry was unsuitable; neither replaces V4.

The untracked 2026-07-14 goal proposes stricter per-color observation gates,
`24/24` development completion, and a newly created post-freeze held-out set.
Those are proposals, not current benchmark authority. Adopting them would
require an explicit dated amendment that supersedes V4 before any sealed
contact; do not silently mix the two protocols.

## Current architecture

The Shared-V5 design uses one online visual encoder for both Camera evidence
and JEPA:

```text
current RGB
    |
    v
shared VisionEncoder
    |------------------------------|
    v                              v
Camera evidence head           BEV lift/decoder
    |                              |
ray hazards, depth offsets,        + action/egomotion
ground-clear logits                |
    |                              v
fixed evidence rasterizer      action-conditioned predictor
    |                              |
unknown/free/occupied BEV           v
                               predicted next latent

next RGB -> EMA target encoder -> EMA target BEV decoder -> target next latent
```

The Camera path resizes RGB to `112x112`, creates a `16x16` grid of
192-dimensional ViT tokens, and builds its dense feature map with a single
non-overlapping stride-seven transpose convolution followed by one `3x3`
convolution. The evidence head predicts:

- 64 ordered first-hit hazards and 64 within-bin offsets per image ray;
- clear/not-clear logits for five ground-support queries per source cell; and
- a fixed differentiable `64x64` unknown/free/occupied raster.

In genuine JEPA training, the predictor should use the current representation
and action to predict the next representation, while an EMA target encoder and
target BEV decoder encode the real next observation. That predictor/target
training did occur in the rejected joint matched-training V4 run. It did not
occur in Camera adaptation V1-V6.

Camera V1-V6 each started from the same Shared-V5 update-zero state produced
by migrating the exact-train-only-qualified N320 encoder/evidence head into a
model whose BEV/predictor side was fresh. They were not continuations of one
another.
They were also not full-from-scratch Camera fits: the encoder/evidence head
began from the N320 migration. Only those two components were then adapted on
the frozen train role.

No state produced by those runs is qualified for G2, navigation, runtime, or
held-out use.

## Data and evaluation state

### Raw supervision

Raw V13 is the terminal valid data audit. Do not rerun or rebuild it merely
because the model failed.

- root:
  `.generated/go2_shared_observable_camera_ray_jepa_v5/development_raw_supervision_v1`
- pairs: `5,172`
- endpoint references: `10,344`
- unique endpoints: `9,460`
- scene shards: `88`
- source files: `354`
- train/checkpoint-selection/probability-calibration pairs:
  `4,262 / 495 / 415`
- all 24 precommitted geometry samples passed across all eight families and
  all three roles
- manifest file/content SHA-256:
  `e102b3c64e99029f118597353966edaaaddbc11efe49b9081d5d7a9c9d974360`
  /
  `74ae5799919ff4d9a06f56d98929cb4cb702d64db52ecdfc93cfa9a8e82fb35a`
- audit file/content SHA-256:
  `0680e1680f30c45feda60498792c3f208c28313e8f087dfbdd1c5807bcf1fe76`
  /
  `0c16e368c9de258d0fbf46e3123d7a3cfcdf60162fd9efa6440d4a7773056aca`

This proves the bytes, role split, inventory, and sampled geometry. It does not
prove that the current architecture can learn the task, but there is no
positive evidence of corruption or a need for another data-refinement cycle.

### Checkpoint-selection population

The Camera and hard-raster decisions use only the fixed development
checkpoint-selection role:

- pairs: `495`
- unique endpoints: `924`
- scopes: aggregate plus eight named families, nine scopes total
- per-scope physical margins: `21`
- total margins: `189`
- wrong-RGB guard: cyclic plus one within family

The nine physical scopes are:

1. aggregate;
2. large enclosed maze;
3. local composite motifs;
4. loop alias stress;
5. medium enclosed maze;
6. open obstacle field;
7. rough local dynamics;
8. small enclosed maze; and
9. visual sensor stress.

No learned checkpoint passed a complete scope. The zero-parameter endpoint
identity oracle passed all `189/189` margins and all `9/9` scopes without
changing thresholds. The gate is therefore attainable; relaxing it is not
justified.

## Experiment chronology

### 1. Small Camera fit and scaling

Camera V11 learned the tiny fit task extremely well and passed 25 of 26 checks:

- raster balanced accuracy: `0.9939025862808951`;
- unknown/free/occupied recall:
  `0.9894560565651553 / 0.9922517022775299 / 1.0`;
- pixel first-hit balanced accuracy: `0.9999430156137219`;
- hit-depth median/p95 error:
  `0.002992570400238037 / 0.010697221755981447 m`;
- ground balanced accuracy: `0.9998668840676225`;
- sole miss: raster NLL `0.07255925759673118 > 0.06`.

V12 added the preregistered gate-aligned raster NLL but was blocked before
execution by an open nested source/proof binding schema. V13-V16 repaired
source binding, open order, visibility, and runtime recovery while keeping the
science fixed. These were governance and runtime corrections, not a sequence
of new model ideas.

The exact-train-only sequential fit ladder reached N320. The first N320 ladder
row failed its numeric gate, after which one compute-scaled N320 successor used:

- seed `20260710`;
- fit size `320`;
- batch size `5`;
- `40,000` updates;
- `200,000` frame exposures; and
- the five equal gate-aligned Camera terms.

That N320 artifact passed `26/26` exact-train-only fit checks. Its bindings are:

- checkpoint file/content SHA-256:
  `ece874b53941e841fffc61b724a86d4383b881549afa453b746dd5d68aba11b0`
  /
  `9dcca536943f89acfd7d463fdab591e19a030ef3dc8f3f19a050b1b10025fc2b`;
- gate file/content SHA-256:
  `4943b4060e88296503c09fc714e55e40fd762527cfccb70a3a341f0df800efe6`
  /
  `76ce5ab703560d171f7c84684b90eed18e8b4cdcc2d8ed3eff6d48496f4de67b`.

What worked: the architecture had enough capacity to fit 320 examples and the
migration was exact.

What this did not prove: disjoint development generalization, a qualified
Shared checkpoint, JEPA utility, G2, navigation, or held-out performance.

### 2. Matched Shared-V5 training V1-V4

The intended experiment trained the promoted Camera+JEPA arm, selected a
checkpoint on the fixed development role, then trained a matched no-JEPA arm
from the byte-identical initialization and schedule.

| Version | What happened | Scientific value |
|---|---|---|
| V1 | Failed before RGB/model forward because absolute development RGB paths did not match the consumer's relative-path contract. | Zero training; infrastructure only. |
| V2 | Path normalization worked, then a valid rank-one scalar was passed to `torch.from_numpy`, which requires an array. | Zero model forwards and zero optimizer steps; infrastructure only. |
| V3 | Completed the first B=4 forward/loss, then strict ROCm determinism rejected `grid_sampler_2d_backward_cuda`. | No completed backward or learned update; infrastructure only. |
| V4 | Used the frozen warning-only exception for the known kernel and completed all 8,000 promoted-arm updates. | Real scientific run; failed the development gate. |

Matched V4 completed:

- `8,000` optimizer steps;
- `8,000` EMA updates;
- `32,000` microbatches;
- `128,000` presentations;
- eight fixed checkpoint evaluations from update 1,000 through 8,000; and
- 72 scope evaluations.

Every one of the eight checkpoints failed all nine physical scopes. No
checkpoint was eligible. The matched no-JEPA arm was never started because the
promoted arm did not pass selection. Therefore:

- there is no qualified Shared checkpoint;
- there is no JEPA-versus-no-JEPA conclusion;
- G2 was not attempted; and
- the V4 state must not be promoted or used as a closest checkpoint.

Terminal audit commit: `73e798aff26b84dc8f7ebc4ff95108c983d761c2`.

### 3. Update-zero transfer and gradient diagnostic

The one-shot diagnostic asked why the exact-train-only N320 Camera fit and
joint Shared training did not transfer.

At Shared-V5 update zero on the disjoint selection role:

- physical scope passes: `0/9`;
- pixel first-hit balanced accuracy: `0.8395965855632654`;
- ground-clear balanced accuracy: `0.8551594601047812`;
- raster balanced accuracy: `0.7189967109838132`;
- occupied recall: `0.36824275623468355`;
- depth median/p95 error:
  `0.2584936320781708 / 2.6021200776100146 m`.

The important gradient findings were:

- Camera fraction of the first-16 joint loss: `0.9900792294092843`;
- Camera global gradient norm: `17,194,663.613836505`;
- JEPA global gradient norm: `0.20507316793061015`;
- Camera-to-JEPA gradient-norm ratio:
  `83,846,481.65992442`;
- joint counterfactual clip factor:
  `5.815757856360626e-8`;
- global Camera/JEPA gradient cosine: `0.007099225591270233`.

Interpretation:

- the migrated Camera was nonzero but inadequate on disjoint development;
- one global gradient clip numerically starved the BEV decoder/predictor;
- the two objectives were nearly orthogonal, not destructively opposed;
- the target representation was not globally collapsed; and
- replacing the predictor was not yet supported by evidence.

This justified separating Camera adaptation from later JEPA training. It did
not authorize a new run by itself.

Terminal audit commit: `a381d9f5160624ff7f092b982ea3b0b479674e86`.

### 4. Protected Camera adaptation V1-V3

These were supervised Camera-only runs:

- trainable: shared encoder, `2,747,520` parameters, 78 tensors;
- trainable: evidence head, `357,993` parameters, 14 tensors;
- frozen: BEV decoder, occupancy head, predictor, target encoder, target BEV
  decoder;
- JEPA objective/backward/EMA: all zero;
- optimizer: separate encoder and head groups with separate clipping.

Each V1-V3 attempt started fresh from the same update-zero migration, not from
the prior failed attempt.

- V1 used encoder learning-rate scale `0.01` and completed 4,000 updates.
  Aggregate raster BA improved from `0.7189967` to `0.8335135`, depth p95
  improved from `2.6021 m` to `1.2168 m`, and ground BA improved from
  `0.85516` to `0.93719`; pixel first-hit BA slightly worsened. All five
  checkpoints failed all nine scopes.
- V2 made the real science change of increasing encoder learning-rate scale to
  `0.1`. It also added immutable metric sidecars and progress observation,
  completed the same 4,000-update boundary, and qualified nothing.
- V3 made the real science change of increasing encoder learning-rate scale to
  `1.0`. It also added guarded, preregistered early-stop controls, completed
  4,000 updates, and qualified nothing.

A proposal to warm-start from V3 was explicitly blocked. Resetting AdamW on
unqualified V3 weights would have been a confounded new attempt and a
closest-checkpoint continuation in disguise.

What worked: finite, stable Camera-only learning; frozen JEPA state remained
unchanged; several physical metrics improved substantially.

What failed: broad disjoint physical qualification. This was not a near miss
and did not license frozen-Camera JEPA training.

### 5. Physical-gate oracle

The preregistered zero-parameter endpoint-identity oracle passed:

- all `189/189` margins;
- all `9/9` physical scopes;
- both frozen evaluators exactly; and
- all wrong-source mappings without a fixed point.

It opened no learned checkpoint and made no learned-performance claim.

What worked: it proved that the literal unchanged physical gate and evaluator
can pass on the frozen population.

What it ruled out: threshold impossibility as the explanation for the learned
failures.

Terminal audit commit: `2de371ed40336eb9415a0afaf5734f8d80ac82a8`.

### 6. Tail-depth Camera V4 and Camera V5

Camera V4 replaced the smooth-L1 within-bin depth term with a tail-depth
p95/CVaR objective. It hit its preregistered numeric progress cutoff at update
1,000:

- passed margins: `97/189`;
- total shortfall: `41.00174362036205`;
- worst margin: `-5.476026201248172`;
- complete scopes: `0/9`.

Updates 2,000 and 4,000 were intentionally not run.

Camera V5 retained the native 8,000-update schedule. Its first exact attempt
failed at update zero because exactly one GPU was not visible. It performed
zero training, evaluation, RGB, or held-out operations and produced no
scientific evidence.

The separately reviewed V5 environment recovery changed only the operational
boundary. It ran to update 1,000 and stopped on its exact cross-run
reproduction floor:

- passed margins: `106`, exactly the required floor;
- total shortfall: `49.13255561472496`, missing the floor by
  `0.033160993206564626`;
- worst margin: `-7.945521640777587`, missing the floor by
  `0.0007632255554206324`;
- complete scopes: `0/9`.

This showed a healthy near-reproduction but not a qualified checkpoint. It
also demonstrated why brittle exact-float reproduction controls should not be
mistaken for scientific progress.

### 7. Final Camera V6

Camera V6 was explicitly the final bounded attempt for the existing
architecture. It:

- started fresh from the same update-zero migration;
- retained the tail-depth objective;
- used same-run coarse progress controls instead of cross-run exact-float
  reproduction;
- completed the full `8,000` updates;
- trained only the encoder and evidence head; and
- performed zero JEPA objective, JEPA backward, or EMA updates.

Progress was:

| Update | Passed margins / 189 | Total shortfall | Worst margin | Aggregate loss | Complete scopes |
|---:|---:|---:|---:|---:|---:|
| 100 | 61 | 112.38103222957155 | -8.20993266105652 | 3.942081972343839 | 0/9 |
| 400 | 84 | 63.34291255226303 | -5.342363524436953 | 2.2573381359346243 | 0/9 |
| 1,000 | 97 | 41.01776266878769 | -5.481336307525642 | 1.797559482710702 | 0/9 |
| 4,000 | 130 | 18.638109619170116 | -2.942538738250731 | 1.4113989913295875 | 0/9 |
| 6,000 | 133 | 16.170685284493313 | -3.231825685501094 | 1.3773113556006658 | 0/9 |
| 8,000 | 135 | 15.360492280690737 | -2.9109309911727883 | 1.3736735407839573 | 0/9 |

The late trajectory had flattened: from 6,000 to 8,000, only two additional
margins passed, total shortfall fell about five percent, loss fell about
0.26 percent, and no scope closed.

What worked in aggregate:

- pixel first-hit balanced accuracy: `0.9771789001258041`;
- ground-clear balanced accuracy: `0.9747100416827836`;
- six distance-group balanced accuracies:
  `0.9688127386413721` to `0.9796585405874831`;
- wrong-RGB sensitivity showed genuine image dependence.

What failed in aggregate:

- raster balanced accuracy: `0.9009460724448773`;
- free recall: `0.91637020862468`;
- occupied recall: `0.8059679976935274`;
- raster NLL: `0.18704089070408247`;
- complete scopes: `0/9`.

Rough local dynamics remained a separate severe failure:

- pixel first-hit BA: `0.8198594673963917`;
- ground-clear BA: `0.647134926562893`;
- depth p95 error: `0.9777327477931971 m`;
- raster BA: `0.7719525130620232`;
- occupied recall: `0.4319466882067851`.

The final update-8000 checkpoint is rejected. It was readable only within the
now-consumed one-shot diagnostic; no further checkpoint access is authorized:

- path:
  `.generated/go2_shared_observable_camera_ray_jepa_v5/protected_camera_adaptation_v6_final_fresh_update0_tail_depth_8k/checkpoints/update_8000.pt`
- file/content/state SHA-256:
  `01871a6495cd6ffa6cdcc97f1451014e887ac9a219360bb69ae0a866db3db20c`
  /
  `4d20f50a688efd617f31ac092a5f7019084afb67e99a064029907222a61be120`
  /
  `960854245db49a048e3a99e91b08d6746795f8c1abd52a267f592900259eee22`.

It must not be promoted, resumed, used as a closest checkpoint, or used for
G2/navigation/runtime.

Terminal audit commit: `f1c4e2efe948165004512ccc1882e721d8626d0b`.

### 8. Bounded architecture postmortem

The postmortem stopped the same-architecture Camera line:

- no Camera V7;
- no longer schedule;
- no seed or learning-rate sweep;
- no loss-weight rebalance;
- no further data refinement;
- no threshold relaxation;
- no closest-checkpoint promotion.

It identified one cheap, causal, zero-training question: whether the fixed soft
evidence decoding and rasterization stage was destroying a large,
cross-scope signal. It also recorded rough-motion perception as a distinct
unresolved upstream problem.

Postmortem commit: `6a67ad77905b44e8a40fa5eef3f8ca7656db349b`.

### 9. One-shot hard-raster diagnostic

The diagnostic preserved the exact V6 raw ray/depth/ground outputs and changed
only the raster conversion:

- finite hit if total finite-hit probability `>= 0.5`;
- depth from the calibrated centre of the maximum-probability finite bin plus
  its predicted within-bin offset;
- ground clear only when in-frustum and the logit is `>= 0`;
- the existing calibrated geometry;
- Boolean five-support and source-cell composition;
- no threshold search, calibration, gradient, optimizer, state mutation, or
  held-out access.

It evaluated 495 selection pairs and 924 unique endpoints once with matched and
cyclic wrong RGB.

Integrity passed:

- exact soft/direct metric reproduction at zero tolerance;
- one checkpoint filesystem read and one deserialization;
- 924 matched and 924 wrong-RGB forwards;
- state hash unchanged;
- zero optimizer, backward, gradient, clip, EMA, checkpoint write, model-state
  mutation, G2, navigation, or held-out operation.

The scientific hypothesis failed:

| Metric | Soft V6 | Hard diagnostic | Change | Required |
|---|---:|---:|---:|---:|
| Aggregate raster BA | 0.9009460724448773 | 0.8716027305278574 | -0.029343341917019927 | +0.05 |
| Aggregate free recall | 0.91637020862468 | 0.951577156783369 | +0.03520694815868897 | +0.05 |
| Aggregate occupied recall | 0.8059679976935274 | 0.6784272740377685 | -0.12754072365575897 | +0.05 |

- scopes with hard-minus-soft BA `>= 0.05`: `0/8`, required `6/8`;
- wrong-RGB sensitivity guard: `8/8`, required `8/8`;
- verdict: `FAIL_HYPOTHESIS_REJECTED`.

The model was using RGB; the hard conversion was simply worse. This rejects
the fixed hard adapter and strongly rejects soft raster conversion alone as a
large recoverable bottleneck. It does not prove that every possible decoder,
upstream representation, spatial resolution, temporal context, or
depth-assisted architecture will fail.

Terminal audit commit: `6e83dc6308e742becf86dbe465cf3650fa85f21a`.

## What worked

1. **Raw-data integrity and role separation.** The build, manifests, role
   counts, inventories, and sampled geometry passed. No current evidence
   justifies rebuilding the data.
2. **Exact-train-only fit capacity.** The N320 Camera fit passed its 26-check
   exact-train-only gate. The architecture can memorize/fit that small
   distribution and its weights can be migrated exactly.
3. **Training infrastructure eventually executed correctly.** Matched V4
   completed 8,000 optimizer/EMA updates; Camera V6 completed 8,000
   Camera-only updates with finite traces and exact frozen-state protection.
4. **The physical gate is attainable.** The oracle passed `189/189` margins
   and `9/9` scopes without threshold changes.
5. **The Camera learns real visual signal.** V6 aggregate direct ray/ground
   metrics are strong, and the wrong-RGB guards show large drops.
6. **The diagnostics prevented another blind run.** The update-zero gradient
   test exposed invalid joint clipping, and the zero-training hard-raster test
   rejected the leading cheap decoder explanation in roughly one forward-only
   pass.
7. **Navigation source plumbing exists.** The hardened development runner
   enforces one RGB/shared forward per tick, reuses cached features for target
   and its source-optional G4 head, validates physical-memory revisions, uses
   frontier/A*, permits at-most-once claims, and publishes immutable
   hash-chained success/fault results. G4 remains mandatory for downstream
   promotion even though the runner can omit that head in earlier stages.

## What did not work

1. **Exact-train-only fit did not generalize.** Passing the N320 fit gate was
   not evidence of disjoint development qualification.
2. **Joint Camera+JEPA training was badly scaled.** Camera gradients dominated
   JEPA by about `8.38e7`, and the one global clip starved the predictor/BEV
   branch.
3. **Separated Camera adaptation did not close a scope.** Multiple fresh
   4,000-update attempts and the final 8,000-update V6 improved individual
   margins but remained `0/9`.
4. **Tail-depth loss was not enough.** It improved the failure profile but did
   not qualify a checkpoint.
5. **Hard rasterization was not the cure.** It reduced aggregate raster BA and
   occupied recall.
6. **Rough motion remains unresolved.** V6's rough-scope ray, ground, depth,
   raster, and occupied metrics were all poor.
7. **JEPA value remains unknown.** The no-JEPA control never ran, and the
   separated Camera line never earned later JEPA training.
8. **Navigation science has not run.** The source-only runner lacks qualified
   model/calibration artifacts and has never been runtime-qualified in a real
   development simulation. Its transitive G2/publication policy closure also
   remains untracked, so the current clean HEAD cannot execute the reviewed
   path by itself.

## What is now ruled out

Without a new architecture-level user decision, do not:

- launch Camera V7 or rerun V1-V6;
- resume or extend the V6 update-8000 checkpoint;
- change only seed, learning rate, schedule length, clip threshold, or loss
  weight;
- rebuild/refine the existing raw data on the assumption that data is the
  problem;
- relax the physical gate or promote the closest checkpoint;
- search raster thresholds or try a second hard raster adapter;
- train the current predictor behind an unqualified Camera;
- run G2, learned navigation, production/runtime, or held-out evaluation;
- spend more time redesigning the reviewed navigation runner.

## What remains scientifically open

The evidence does not currently distinguish among:

- insufficient spatial/multiscale decoding between the `16x16` token grid and
  dense evidence;
- missing temporal or attitude context, particularly in rough motion;
- upstream representation invariance/generalization limits;
- a direct learned BEV output contract instead of the current ray/ground
  evidence contract;
- a depth-assisted perception/runtime claim instead of RGB-only;
- whether JEPA adds value after perception is actually qualified; and
- navigation performance once valid artifacts exist.

Do not combine these into one large successor. Select one mechanism.

## Navigation runner state

The initial runner at `e30020c87ee30e80b1a9bcabea94930e47429cc6`
was correctly blocked on:

- missing artifact cross-binding;
- incomplete claim/evidence publication;
- incomplete reset/revision/fault lifecycle; and
- missing enforcement of the development scene role.

Commit `540c5865d9a190b509e1e3b45c962a2cd6159b3f` fixed those findings. A
different agent returned `PASS_SOURCE_ONLY`; 12 accelerator-hidden tests passed
in `0.69 s`. The reviewed files remain byte-identical:

| File | SHA-256 |
|---|---|
| `lewm/navigation/genesis_shared_v5_dev_stack.py` | `5f596bf78bbc2e95a08e379317a15b1d7fb946ba06a43f984be49864e5b41183` |
| `lewm/navigation/shared_v5_dev_runtime.py` | `000bd16f34967e4c4158fb7ad62723dd100dcb3a935eac213515d5ed6f5f83c3` |
| `lewm/tests/test_shared_v5_dev_runtime.py` | `bdd03bb0650e9d3ee6e73a01ea3a6d0723a860ff8c06bd3bf0f2cea24193b1f2` |
| `scripts/run_go2_shared_v5_dev_maze.py` | `ec5f24531d6d875c22c29b1b56bb5c4d5d000d263160ac66ec30e3f86cce0c06` |

That `PASS_SOURCE_ONLY` does not establish clean-HEAD import or execution
closure. The runner imports
`lewm/benchmarks/go2_shared_jepa_v5_full_training_v4_policy.py` and
`scripts/go2_shared_jepa_v5_one_shot.py` at module load, and both are currently
untracked. The associated G2/finalization boundary also currently depends on
these untracked sources:

- `lewm/benchmarks/finalize_shared_observable_camera_ray_jepa_v5_g2.py`;
- `lewm/benchmarks/shared_observable_camera_ray_jepa_v5_finalizer_core.py`;
- `lewm/benchmarks/shared_observable_camera_ray_jepa_v5_runner_policy.py`;
- `lewm/models/shared_observable_camera_ray_jepa_v5_authority.py`; and
- `lewm/models/shared_observable_camera_ray_jepa_v5_registry_policy.py`.

Reuse the reviewed design, but independently review and commit this minimal
transitive boundary, or a tracked equivalent, before claiming that a clean
checkout can publish G2 or invoke the development runner. Do not sweep the
other dirty research tree into that change.

The runner is a real one-development-maze integration runner, not a held-out
launcher or physical-locomotion qualification. It still requires:

- a non-synthetic production-eligible G2 PASS publication and final report,
  cross-bound to the Shared checkpoint;
- qualified physical-head/camera calibration;
- a fixed, source-bound, load-valid target head and calibration on every
  invocation, including the one-scene/G3 smoke; its target-conversion science
  is qualified later at isolated G5;
- the learned G4 head and calibration before downstream G5/G6 promotion;
- an authorized visible-development scene/platform/observer bundle;
- an immutable output custody destination; and
- separate execution and runtime-qualification authority.

Reuse this runner. Do not add frameworks, resume machinery, plugin systems, or
generic logging.

## Lean restart plan

The following is a proposal for the next user decision and future
preregistration. It is not execution authority.

### Step 1: choose the claim and one architecture mechanism

Choose one:

1. **RGB-only genuine-JEPA path.** Authorize exactly one materially different
   perception successor. This preserves the scientific goal.
2. **Depth-assisted path.** Change the claim and runtime contract explicitly,
   then qualify depth-assisted occupancy. This may be easier operationally but
   is not the same RGB-only result.
3. **Stop the line.**

If RGB-only remains the goal, the leanest first candidate is a genuinely
multiscale spatial evidence decoder that replaces the current single
stride-seven dense lift while preserving the same ray/depth/ground evaluator
contract. It directly targets an open architectural weakness, keeps the data,
metrics, geometry, Shared model interface, and downstream runner intact, and
is cheaper to falsify than adding spatial changes, temporal memory, and a new
output contract together.

Temporal/attitude context is a separate candidate motivated by the rough
motion failure. Do not combine it with the multiscale change in the first
probe. A direct-BEV or depth-assisted output changes the qualification contract
and must be decided explicitly before source work.

### Step 2: source-only feasibility

Before any GPU run:

1. write a short architecture decision and preregistration naming exactly one
   changed mechanism;
2. reuse the existing Raw V13 train/selection/calibration roles, physical
   evaluator, geometry, wrong-RGB mapping, and reviewed runner;
3. freeze the parameter budget, output contract, optimizer, checkpoints,
   maximum presentations, and stop rules;
4. close, independently review, and commit the minimal untracked
   G2/publication/runner dependency boundary;
5. run shape, finiteness, source-closure, migration, frozen-state, and
   synthetic microfit tests;
6. use a tiny overfit only to catch wiring/capacity failure, never as evidence
   of generalization; and
7. obtain a different-agent architecture source review.

Do not build another audit framework. One preregistration, one implementation
handoff, one independent review, and one terminal result are sufficient.

### Step 3: one cheap development falsification

This is a proposed user-chosen continuation heuristic, not an existing
committed gate. Use one fixed seed and one attempt, capped at both the
V6-equivalent `16,000` training presentations and a preregistered compute
budget. Observe at equivalent `1,600`, `6,400`, and `16,000` presentations if
batching remains comparable; otherwise preregister the presentation mapping
before launch. Do not use update count alone when architecture or batching
changes. No calibration search, data rebuild, threshold change, or second
seed.

For a successor that preserves the current physical output contract, a
reasonable aggressive proposed continuation gate at the final cheap-probe
boundary is:

- at least `1/9` complete physical scopes, versus V6's `0/9`;
- passed margins strictly above the V6-equivalent `97/189`;
- total shortfall strictly below the V6-equivalent
  `41.01776266878769`; and
- all three rough-motion direct indicators strictly better than V6:
  pixel BA `> 0.8198594673963917`, ground BA
  `> 0.647134926562893`, and depth p95
  `< 0.9777327477931971 m`.

Failure stops that architecture. Do not change a seed or continue because the
loss looks promising.

### Step 4: only a passing probe earns a bounded qualification run

Only the Step 3 pass can earn a separately preregistered qualification run.
Fix its presentation count and compute cap, and allow at most one
predeclared intermediate cutoff using equivalent-presentation V6 baselines.
Do not invent a `3/9`, then `6/9` progression: those thresholds are not
supported by the committed contract. The final gate remains unchanged at
`9/9` scopes and `189/189` required margins.

Missing the predeclared cutoff stops. A passing loss curve alone does not
license consuming the remaining budget.

### Step 5: train JEPA only after perception qualifies

After a Camera/perception checkpoint passes:

1. freeze the qualified perception boundary;
2. train the BEV decoder/action-conditioned predictor with its own optimizer
   and clipping boundary;
3. maintain the target encoder/target BEV EMA contract;
4. select/calibrate only on the existing fixed development roles;
5. publish a strict pre-G2 candidate, not a post-G2/runtime checkpoint; and
6. run a matched no-JEPA control from the identical initialization and
   schedule only if the intended result explicitly makes a causal JEPA claim.

Do not repeat the rejected single global Camera+JEPA clip.

### Step 6: untouched G2

The committed G2 metric and dataset-role contract exists, but a runnable
Shared-V5 G2 admission/finalization path is not closed in committed source.
Before any untouched-G2 access, land the independently reviewed minimal source
closure named above and rerun its accelerator-hidden contract tests. Then run
G2 once for the selected Shared checkpoint. The gate requires at least:

- planner-admitted free precision `>= 0.99`;
- obstacle recall within 2 m `>= 0.95`;
- useful traversable-space recall `>= 0.90`; and
- routes on predicted maps not exceeding the oracle-map collision gate.

Failure blocks navigation. G2 is not a tuning set.

### Step 7: bind and qualify the reviewed runner in gate order

Before the first invocation, bind the G2-qualified Shared checkpoint, physical
calibration, and the runner-required fixed target head/calibration. The target
artifact must load and remain fixed for G3, but its scientific
target-conversion qualification is deferred to isolated G5. The G4 artifact is
optional for the early G3 invocation and mandatory before downstream
promotion. Keep development authority and output custody bound throughout.

Progress cheaply:

1. rerun its source-only contract suite if any dependency binding changes;
2. one authorized visible-development scene smoke;
3. G3 on the fixed 600-tick fast-development panel;
4. mandatory G4 offline ranking, deterministic-information-gain comparison,
   and DAgger gap closure;
5. isolated G5 target conversion under oracle coverage;
6. G6 on the full 24-scene, 2,400-tick development suite; and
7. G7 locomotion, odometry, noise, action-source, and physical-smoke
   robustness.

Do not start full-scene sweeps while an offline or one-scene gate is failing.

### Step 8: freeze and execute the existing V4 sealed role

Only after all development gates pass:

1. freeze every source, checkpoint, calibration, threshold, seed, geometry,
   reset rule, simulator setting, evaluator, scorer, environment, and output
   schema;
2. bind the freeze to the existing V4 sealed commitment and its one-shot
   launcher/finalizer;
3. preserve independent custody and expose only the commitment, count, and
   one-shot interface until execution;
4. materialize and open the V4 sealed role only through that final authorized
   execution; and
5. publish the result once with no retry or tuning.

Creating a replacement held-out population is not part of the current
contract. It requires a dated amendment that explicitly invalidates or
supersedes V4 before any sealed contact.

## Anti-overengineering rules

- One hypothesis, one architecture, one seed, one attempt until a named gate
  passes.
- No more raster diagnostics or threshold search.
- No data polishing without a preregistered learning-curve or corruption test
  that actually indicates a data problem.
- No full training after only source tests or local microfit success.
- No full 24-scene navigation while a one-scene or small-panel gate fails.
- Reuse current data, geometry, role splits, physical evaluator, belief maps,
  G3/G4/G5 source foundations, and the reviewed runner.
- Do not add a generic training, logging, plugin, resume, or audit framework.
- Publish only the fixed checkpoint summaries needed to decide stop/continue.
- Never use a held-out result for tuning, selection, calibration, repair, or
  successor choice.

## Repository and process state

At the start of this handoff:

- current HEAD: `6e83dc6308e742becf86dbe465cf3650fa85f21a`;
- worktree: 38 modified tracked files and 569 untracked files;
- deletions/renames/copies: zero;
- the current reviewed runner files and the Camera diagnostic closure are
  clean at their bound hashes;
- no matching repo training, pytest, benchmark, render, navigation, or
  diagnostic process was observed;
- no GPU query was made for this handoff, so this document does not claim
  hardware-idle proof;
- no experiment, dataset, RGB, navigation, or held-out payload was opened to
  write this handoff; docs, source files, and Git metadata were used.

The dirty worktree is pre-existing research work. Do not run `git clean`,
`git reset --hard`, `git checkout --`, or broad restore commands. Do not stage
or commit unrelated files. The older goal and 2026-07-14 handoff are themselves
currently untracked local evidence; preserve them.

The governing generalization-contract path is also modified in the pre-existing
worktree. This handoff cites its committed HEAD bytes, SHA-256
`316766b87994ba70994828559ba8cf80d33c1d11a14299c0d3516b66b663d06a`.
The dirty edit does not supersede the committed V4 contract unless it is
reviewed and committed as an explicit dated amendment.

## Authoritative evidence index

| Evidence | Commit | File SHA-256 | Canonical content SHA-256 |
|---|---|---|---|
| Governing G0-G8 contract, committed HEAD bytes | `6e83dc6308e742becf86dbe465cf3650fa85f21a` | `316766b87994ba70994828559ba8cf80d33c1d11a14299c0d3516b66b663d06a` | n/a |
| Matched V4 terminal numeric audit | `73e798aff26b84dc8f7ebc4ff95108c983d761c2` | `70371a2cd09e912e05ba0b5efdf75ee2de38cc89347e8111fff303e2a55c485b` | `ae86d1479fc3016eb96302304e079b7bf9647e26b24b3d860e7d32013bf9c6f4` |
| Update-zero transfer/gradient audit | `a381d9f5160624ff7f092b982ea3b0b479674e86` | `52d6ac4a7287b9cb9bd33fbdd3eadbb939f9368643953fe61866777f620914bf` | `86276a89a6cc637aefdee798c916eeaafe1d30ca2c69038b6debdaa8332f8fe0` |
| Physical-gate oracle audit | `2de371ed40336eb9415a0afaf5734f8d80ac82a8` | `a899c199cb03be09c795eac0203747e6e1e507cca6d2e4f5a5b9db3b41435dee` | `29f406b6bfa251819dff7e56c69adc8dbe2244037e24594d422156780f14d617` |
| Final Camera V6 terminal audit | `f1c4e2efe948165004512ccc1882e721d8626d0b` | `367dd08f9a039710d61efd9ecb652134f6efbd056e126c4a51d67929f28b06b7` | `76727ada6442774412508b0ca96b1a50b5170bc75867235aecc132f28d1ac892` |
| Camera V6 architecture postmortem | `6a67ad77905b44e8a40fa5eef3f8ca7656db349b` | `7f5ca06e773c61b24fe792f210c38204066c35ee7ebd496e5a75174b9933d0b1` | n/a |
| Hard-raster terminal audit | `6e83dc6308e742becf86dbe465cf3650fa85f21a` | `c25fdf3e2f33457555ba2ef10a83cfb67184f4d2471fb2a8fc0cfc3c26bc148c` | n/a |
| Development-runner source review | `540c5865d9a190b509e1e3b45c962a2cd6159b3f` | `8157780646cebe37301c14e3ef1fbf5139216e5918d8ce67ea1ad825932c94c6` | `4a1e237e6bc2433e57c42570572111b985bec550996e0a786dd050c3a3b5f832` |

Primary paths:

- `docs/lewm_go2_generalization_execution_contract_2026-07-09.md`
- `docs/lewm_go2_shared_jepa_v5_matched_training_v4_terminal_numeric_failure_audit_2026-07-15.json`
- `docs/lewm_go2_shared_jepa_v5_update0_transfer_gradient_diagnostic_v1_terminal_audit_2026-07-15.json`
- `docs/lewm_go2_shared_jepa_v5_protected_camera_physical_gate_oracle_v1_terminal_audit_2026-07-15.json`
- `docs/lewm_go2_shared_jepa_v5_protected_camera_adaptation_v6_terminal_audit_2026-07-23.json`
- `docs/lewm_go2_shared_jepa_v5_camera_v6_bounded_architecture_postmortem_2026-07-23.md`
- `docs/lewm_go2_shared_jepa_v5_camera_v6_hard_raster_diagnostic_v1_terminal_audit_2026-07-23.md`
- `docs/lewm_go2_shared_v5_dev_runner_independent_successor_review_2026-07-15.json`
- `docs/lewm_go2_heldout_maze_goal_2026-07-14.md`
- `docs/lewm_go2_ready_to_benchmark_handoff_2026-07-14.md`

Read the generalization contract from committed HEAD, not from the dirty
worktree copy. The final two paths are untracked local context and do not
supersede committed authority.

## Exact restart order

1. Read this handoff, the V6 terminal audit, architecture postmortem,
   hard-raster terminal audit, and runner source review. Do not reopen the
   rejected checkpoint or diagnostic output merely to reconstruct context.
2. Confirm the branch and preserve the dirty worktree. Do not clean or reset.
3. Make the explicit RGB-only versus depth-assisted versus stop decision.
4. If RGB-only, select one spatial or temporal mechanism, not both.
5. Write the lean fixed-presentation/fixed-compute falsification contract and
   freeze it before implementation.
6. Implement only the named mechanism, close the minimal untracked
   G2/publication/runner source boundary, run accelerator-hidden/source tests,
   and obtain an independent source review. Do not launch a full run.
7. Run the one cheap probe only after exact authorization.
8. Scale only if its frozen continuation gate passes.
9. Train/separate JEPA and attempt the source-closed G2 once only after
   perception qualification. Add the no-JEPA arm only for an explicitly
   retained causal JEPA claim.
10. Bind the required fixed target artifact and existing development runner;
    pass G3, mandatory G4, isolated G5, full G6, and G7 in order rather than
    redesigning it.
11. Freeze and execute the existing V4 sealed role once only after every
    development gate passes.

## Authority boundary

This handoff records evidence and proposes a restart sequence. It does not
authorize:

- a new architecture implementation;
- a Camera or JEPA training run;
- a retry, resume, threshold search, or checkpoint promotion;
- any further access to rejected checkpoints; the sole bound diagnostic read
  is consumed;
- G2, navigation, runtime, production, deployment, hardware, or held-out
  execution; or
- replacement, materialization, or opening of the V4 sealed role.

All previous one-attempt roots are consumed and immutable. The next
state-changing step requires a new user architecture decision, a narrow
preregistration, a different-agent source review, and explicit execution
authorization.
