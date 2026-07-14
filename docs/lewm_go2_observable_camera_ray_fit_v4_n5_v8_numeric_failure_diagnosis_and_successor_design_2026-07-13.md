# Camera-ray N5 V8 numerical failure diagnosis and successor design

Date: 2026-07-13

Status: **read-only diagnosis and design proposal; no execution authority**

## Scope and immutable evidence

Camera V8 repaired the verifier process lifecycle and completed the sole exact
seed-`20260710`, N=5 attempt. The attempt is terminal and may not be retried,
edited, used as a checkpoint, or treated as a later-rung result.

The diagnosis below uses only the canonical V8 result, independently recomputed
metric receipt, gate, and already-reviewed source. It does not infer a missing
metric, open another role, or change a threshold.

| Evidence | File SHA-256 | Content SHA-256 |
| --- | --- | --- |
| Independent source review | `fd095eea8b1f2a0cde67f77a3bd2338f8f13e3a81d824777475600a258758f0f` | `b83b571331e428d3db46462567bf05e23d9b37909b42fda89e4efeb38baca81d` |
| Metric verification | `b28cbd3795d090d652504a4721216689f160577e7947e3671b04688e39ae6b89` | `c3bf90bc16bff983232d9a23de20a881637233e5e3b4723f5134769a2d5c7090` |
| Gate | `cfe39b64e496bbd7bf4a2b0144bffee884c9b2ceca18d1d5275f41492633c081` | `11f02aa3fb51b217d4b2a18544582f42f4593a44c01b28cd733df4f6873f4ddf` |

The verifier recomputed every evaluation loss, confusion matrix, depth
quantile, family metric, wrong-RGB control, and threshold. It loaded the final
checkpoint only for metric verification, used only the five selected train
RGBs, used GPU0 (`AMD Radeon AI PRO R9700`), and recorded zero G2, held-out,
selection, calibration, runtime, hardware, production, or GPU1 opens.

## What passed and failed

The gate passed 19 of 26 checks and failed seven.

| Quantity | V8 value | Frozen requirement | Result |
| --- | ---: | ---: | --- |
| Ground-clear overall balanced accuracy | `0.9972425204` | `>=0.99` | PASS |
| Ground-clear distance bins | `0.9958664` to `0.9996299` | each `>=0.97` | PASS |
| Ground-clear scene families | `0.9807288` to `1.0` | each `>=0.97` | PASS |
| Pixel hit/no-hit balanced accuracy | `0.6633690955` | `>=0.99` | FAIL |
| Hit-depth median absolute error | `0.0429596305 m` | `<=0.06 m` | PASS |
| Hit-depth p95 absolute error | `0.5386402130 m` | `<=0.15 m` | FAIL |
| Raster NLL | `1.0748962879` | `<=0.06` | FAIL |
| Raster balanced accuracy | `0.5617928196` | `>=0.99` | FAIL |
| Raster unknown recall | `0.6846740681` | `>=0.97` | FAIL |
| Raster free recall | `0.0007043907` | `>=0.97` | FAIL |
| Raster occupied recall | `1.0` | `>=0.97` | PASS |
| Wrong-RGB raster balanced-accuracy drop | `0.0059974709` | `>=0.08` | FAIL |

The pixel confusion matrix is decisive. Of 26,323 no-hit rays, 17,721 were
predicted as hits; of 20,717 hit rays, 20,716 were predicted as hits. The model
therefore predicts a hit for about 81.7% of rays when the target rate is about
44.0%. This is not a generic failure to see RGB: wrong RGB causes large and
passing degradation in pixel balanced accuracy (`0.1911`), median depth
(`+0.4661 m`), p95 depth (`+4.6219 m`), ground accuracy (`0.3987`), and raster
NLL (`+1.7435`). It is a badly placed hit/no-hit decision boundary plus a tail
depth error.

The raster failure is mostly downstream of that boundary. The target raster
contains 16,123 unknown, 4,259 free, and only 98 occupied cells. V8 recalls all
98 occupied cells but predicts occupied for 5,084 unknown cells and 4,256 free
cells. Only three free cells are predicted free. The wrong-RGB raster
balanced-accuracy control cannot separate when both matched and wrong RGB
collapse to nearly the same thresholded occupied-heavy output, even though the
probabilistic raster NLL does separate strongly.

## First-principles cause

### 1. The optimized pixel loss does not match the gate's class balance

`ordered_obstacle_first_hit_nll_breakdown_v4` gives one equal-weight group to
all no-hit rays and one equal-weight group to every represented hit-distance
bin. With `G` represented hit bins, no-hit evidence receives `1/(G+1)` of this
loss and hit evidence receives `G/(G+1)`. The gate instead weights hit recall
and no-hit recall equally.

This skew-resistant depth-bin loss is useful inside the hit class, but it is
not a balanced hit/no-hit objective. The exact failure direction is the one the
mismatch predicts: nearly perfect hit recall and poor no-hit recall.

### 2. Raster occupancy inherits the pixel false positives

The differentiable raster marks occupancy from integrated pixel first-hit
probabilities before separating free from unknown using ground support. A
large excess of predicted hits therefore turns true free and unknown cells
into occupied cells. The ground branch can pass every direct classification
check and still be masked by erroneous occupancy, exactly as V8 demonstrates.

### 3. The tail is not converged

Median hit depth passes, while p95 error is 3.59 times its limit. The stored
training trace is still improving at the final update: ordered first-hit NLL
falls from `0.8201` at update 360 to `0.7453` at update 400, raster loss falls
from `0.3013` to `0.2905`, and the final total is the best stored total. There
is no plateau evidence supporting a hard capacity limit. Four hundred updates
are insufficient evidence for either convergence or architectural incapacity.

### 4. N=5 cannot establish scene generalization

The panel contains exactly one frame from each of five families. It is an
intentional memorization/observability test. Passing it can show that the
output, loss, and model can represent five registered views; it cannot show
novel-scene generalization. N16/N32/N320 and scene-disjoint V5 roles remain
necessary after, and only after, the fit gate passes.

## Narrow additive successor

The next experiment should change the objective before changing capacity. It
must use a new namespace, a pre-implementation amendment, frozen source,
different-agent review, one fresh attempt, and the unchanged N5 metric gate.
It is not a V8 retry.

### Hierarchical first-hit objective

Keep the V4 target, calibration, model output, physical rasterizer, five-frame
panel, wrong-RGB control, and all 26 thresholds unchanged. Replace only the
first-hit loss with a metric-aligned hierarchy derived from the existing
normalized ordered distribution:

1. `presence`: equal-weight mean NLL for target no-hit rays and target hit
   rays, using `P(no_hit)` and `sum_d P(hit_at_d)`;
2. `conditional_depth`: equal-weight mean NLL over represented depth bins,
   using `P(hit_at_d | hit)` only on target hit rays; and
3. `hierarchical_first_hit = 0.5 * presence + 0.5 * conditional_depth`.

This retains ordered, normalized physical probabilities while separating the
binary decision required by the gate from depth resolution inside the hit
class. The four top-level V4 terms remain equally weighted: hierarchical
first-hit, within-bin offset, ground clear, and derived raster.

### Convergence contract

The successor should preregister deterministic full-panel training for at most
4,000 updates, with immutable diagnostic snapshots every 100 updates. It must
select the final update only; it may not choose a checkpoint after seeing gate
metrics. The longer bound is justified before execution by the monotone V8
trace and remains small on the R9700. Exact verification must again run in the
reviewed fresh isolated child.

The successor passes only if all existing N5 checks pass. A failure remains
terminal and cannot be repaired by threshold, calibration, post-processing,
checkpoint selection, or retry.

### Capacity escalation only if localized

Do not add a larger encoder or high-resolution skip path in the first
successor. If the hierarchical-loss successor passes pixel presence and raster
checks but still fails only p95 depth, that result can justify a separately
reviewed local-detail decoder successor. If hit/no-hit still fails, the
hierarchical formulation or optimization is wrong. If presence passes but the
raster remains occupied-heavy, the physical raster coupling must be audited
before changing model capacity.

## Falsifiable predictions

| Hypothesis | Prediction under the hierarchical-loss successor | Falsification |
| --- | --- | --- |
| Loss/gate mismatch drives false hits | No-hit recall rises from `0.3268` and pixel balanced accuracy approaches the frozen gate | Hit recall remains near one while no-hit recall remains low after the presence term converges |
| Raster failure is downstream of false hits | Unknown/free recall and wrong-RGB raster BA separation rise when pixel false positives fall | Pixel presence passes but matched raster remains occupied-heavy |
| Current model has enough N5 capacity | All training components continue down and five-frame metrics pass within the frozen 4,000-update bound | Metric-aligned losses plateau above the gate despite correct gradients and deterministic optimization |
| Tail error is mainly undertraining | p95 depth declines materially while median stays within threshold | Presence and conditional-depth losses converge but p95 remains above `0.15 m` |

## Downstream consequence

Raw-supervision V7 and full-training source implementation can continue because
they do not consume the failed checkpoint. Canonical raw construction may run
only after its own dual review/authorization chain. Exact shared-JEPA training
must remain blocked until a camera successor qualifies a migration checkpoint
or a separately preregistered shared-model successor removes that dependency.
G2, held-out navigation, runtime, hardware, production, and promotion remain
sealed.
