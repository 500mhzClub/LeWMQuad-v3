# RGB/body ground projection: completed result and navigation consequence

All8 preserved multi-junction routes and1,498 actual RGB/body packets completed
the [fixed ray-envelope diagnostic](go2_ground_projection_envelope_development_v1_2026-09-05.md).
There were no sensor failures, missing frames, retries or conditional interval
enclosure failures. The4 original task failures remain in the population.
Session66047 terminated exit0; its new5 source/test/protocol paths are now bound
and must not be edited or rerun. No training, new physics or policy selection
occurred in this study.

Root: `.generated/go2_ground_projection_envelope_development_v1_attempt_001`.
Result SHA: `c31d1c3e9d747d9a03e396286f47aad47e2dd6bd9f10106e3e7d6cf7ef7464a9`.
Launch SHA: `47923287b16fc6e9bfb2c33d05f821d1c1e90ff760c72224305cab19671f42f3`.
The139 source and141 input bindings were reverified unchanged after completion.
The19 new synthetic geometry/accounting tests passed before execution. These
checks establish the stated ray calculation, not robot clearance or calibration.

## Point errors reveal a limit hidden by body-height averages

Across7,190,400 sampled pixel rays,1,358,287 see actual visible floor. None of
these floor rays has optical depth below0.5 m; this is a property of the sampled
recorded views, not a universal camera guarantee. Current-frame evidence does
not cover the nearby ground under the robot. There are no visible-floor samples
beyond4 m, so these recordings also do not establish distant-floor performance.

Nominal projected point error on actual visible floor averages0.04941 m and
reaches1.18039 m. Counts exceeding0.05/0.1/0.25/0.5 m are430,361/153,651/26,036/
3,441. These are correlated pixel/frame counts, not independent trials. Per-route
means range0.02907–0.09662 m.

| Actual optical depth | Visible-floor rays | Mean point error | Maximum point error |
|---|---:|---:|---:|
| 0.5–1 m | 1,047,853 | 0.04351 m | 0.78536 m |
| 1–2 m | 299,698 | 0.06863 m | 1.18039 m |
| 2–4 m | 10,736 | 0.08785 m | 0.43618 m |

The largest error is not a hypothetical infinite-distance horizon artifact:
it occurs at an actually visible1–2 m floor point in the1.2-m-wide dead-end
return route. At that frame (index207,22.2 s), the existing ground estimator has
normal error0.08095 rad, body-height error0.02445 m and camera-height error
0.04378 m. Initial normal error was only0.000539 rad in every route, while
terminal normal errors range0.00962–0.07973 rad. This points to accumulated
attitude error and its camera-lever-arm/ray amplification as a useful next
diagnostic, not a demonstrated single cause or justification to refit on this
frame. The old gyro/body source and all old results remain unchanged.

## Conditional coverage is not useful precision or safety

Bottom-connected RGB evidence marks1,357,353 pixels:1,355,532 are true visible
floor and1,821 are false-surface positives. Every sensitivity family accepts
these same pixels in this dataset; none requires abstention on the positive
visible-floor subset. Near-horizon abstention is implemented/tested but not
triggered by this subset. All1,821 false-surface acceptances remain reported;
a geometrically bounded hypothetical ground point does not make the actual
observed surface floor.

| Height radius / normal radius | Frames whose true plane is inside hypothesis (of1,498) | True-floor intervals missing actual range (of1,355,532) | Mean / maximum interval width |
|---|---:|---:|---:|
| 0.01 m / 0.025 rad | 995 | 113,405 | 0.254 / 2.070 m |
| 0.01 m / 0.05 rad | 1,258 | 34,566 | 0.461 / 4.807 m |
| 0.01 m / 0.10 rad | 1,279 | 0 | 0.973 / 78.942 m |
| 0.03 m / 0.025 rad | 995 | 49,936 | 0.368 / 2.475 m |
| 0.03 m / 0.05 rad | 1,387 | 25,098 | 0.578 / 5.301 m |
| 0.03 m / 0.10 rad | 1,498 | 0 | 1.101 / 83.097 m |
| 0.05 m / 0.025 rad | 995 | 28,417 | 0.482 / 2.880 m |
| 0.05 m / 0.05 rad | 1,387 | 11,344 | 0.694 / 5.795 m |
| 0.05 m / 0.10 rad | 1,498 | 0 | 1.229 / 87.252 m |

Widths are3-D distances along each calibrated ray. Bounds separate correlated
normal terms and can be loose. Even the smallest family yields no accepted
interval of width at most0.1 m. For the0.03 m/0.10 rad family, only50,310 of
1,355,532 intervals have width at most0.5 m. No radius is selected or calibrated
from this table. Empirical coverage under a loose hypothesis is not a confidence
level, footprint guarantee, or robust transfer result. Plane, camera and palette
assumptions remain substantive limitations.

## Next implementation, tied to actual navigation

1. Improve and separately test causal attitude/ground estimation before using
   metric floor endpoints as place or clearance evidence. Compare the unchanged
   gyro-only baseline with a fixed gravity-feedback estimator using actual
   specific-force history transported into a common body frame. Gate unreliable
   force information and report failures under acceleration; acceleration and
   gravity are not independently observable from one instantaneous force vector.
   Retain fixed parameters and all original traces, then test fresh excitation
   before claiming accuracy or hardware validity. Do not change the running JEPA
   training study or its sensor tensors during this comparison.
2. Add temporal observation retention and explicit unknown/occluded space. The
   forward camera's current visible floor omits the near-body region; previously
   observed evidence needs motion transport and growing uncertainty. Do not fill
   that gap with an imagined free corridor or a privileged floor map.
3. Advance an actual exit-observation/active-scan component and independently
   measured arrival events. A direction with visible floor is not yet a unique
   exit, a revisited place, or a qualified directed edge. Preserve ambiguity and
   require real executed traversal evidence before changing memory connectivity.
4. Address RGB appearance dependence separately. The palette baseline failed
   grayscale and channel-swap controls; better geometry does not repair that
   limitation. Compare appearance-robust/learned observations under fixed data
   splits and controlled geometry/appearance changes, then connect them to the
   memory/controller bridge.

The18-model context-matched JEPA/data-coverage comparison continues separately.
Its final audited contrasts decide the next fixed learned-control experiment.
Neither those offline metrics nor this projection result prove exploration,
beacon discovery, remembered-goal return, fresh-maze generalization or hardware
success. The original scientific objective remains active.
