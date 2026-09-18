# Chained tracking plus single-pass composition prepared, not adopted

The active native attempt uses `MeasuredPlaneChainedAnchorController` and
explicitly excludes the separately verified single-pass optimization. A new
source-only composition, `MeasuredPlaneChainedSinglePassController`, combines
the existing optimized measured-plane controller with the exact existing
`MeasuredPlaneChainedAnchorVisualMotion` implementation. Neither original
controller, observer, live launcher, protocol nor bound test is modified.

The composition retains the verified body-projection, geometry, bounds and
receipt handling of `MeasuredPlaneSinglePassController`. Its observer is the
same concrete class as the active chained controller, including measured-plane
refinement around original/direct/chained image fits, missingness handling,
conflict rules, retained references and the existing bridge budget. The new
controller publishes an explicit composition identity and the same two chained
tracking flags.

The separate comparison helper validates those tracking flags and reuses the
existing performance normalization. It retains complete tracking receipts,
decisions and unexpected scientific fields. Complete observed-state comparison
uses the existing full-history state helper, including visual motion and
mission state; it does not normalize their implementation types.

All four actual synthetic image-to-action integration cases passed in 53.34
seconds:

- JEPA/full with ordinary descriptor associations.
- Direct/no-RGB with ordinary descriptor associations.
- Direct/no-RGB with descriptor associations removed, exercising actual
  primary-camera chained flow, endpoint geometry and measured-plane refinement.
- Direct/no-RGB with blank primary images and descriptor associations removed,
  exercising the corresponding actual auxiliary-camera chained fit.

In each case the original chained and new composed controllers had equal
complete normalized decisions and observed-state fingerprints after every
observation. Both chained cases admitted a real chained fit with plane
refinement and preserved original inliers. A duplicate final packet exercised
the same sensor-failure latch and zero command. Each controller made exactly
one actual synthetic model forward; model tensors remained unchanged and no
parameter gradients appeared. Changed tracking flags are rejected, and an
unexpected scientific field remains visible to comparison.

| Prepared file | SHA-256 |
| --- | --- |
| `lewm/measured_plane_chained_single_pass_controller_development.py` | `b3e77645602f8e3ba5ba40ffbbb00440e79fb6053c152d7603a3fc1360b6d786` |
| `scripts/measured_plane_chained_single_pass_comparison_development.py` | `78abf551f2d6a4ecd2df9c4f1e20a6570bb9e2aede9c4e9224bca10879a055de` |
| `lewm/tests/test_measured_plane_chained_single_pass_controller_development.py` | `74b5caae9f7b9df9fd06f1b73cbbad8cc2e49e326c48875e85c9200e23f4b778` |

These small synthetic tests do not establish long-history equivalence, a
latency reduction for this composition, or navigation recovery. The previously
measured 53.1218% reduction belongs to the non-chained single-pass replay and
must not be transferred to this composition. Before adoption, compare complete
decisions, actual forecasts and retained state on authenticated recorded
history with fresh independent assigned models, including the real chained
reacquisition and return behavior. Any component timing must preserve the
actual controller implementation and report instrumentation overhead.

No recorded-history replay, new native attempt or model training was launched
by this preparation. The original native parent PID 2992412 and worker PID
2994743 remain the active experiment under launch
`0f2963636abc4f1fc738e9323d2a74db02888ca7062dc1afa21023f8c443b8ff`.
That worker was still in startup validation when this note was written. The
full navigation, independent-maze, timing and hardware goal remains incomplete.
