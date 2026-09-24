# Tracking recovery: retained-anchor loss at frame 863

The fresh no-RGB JEPA direct-flow collection stopped at frame 863 after exhausting
the ten-frame measured bridge allowance. This is a diagnosis of the closed
collection, not a completed native audit or verified recovery result.

The complete 874-row decision stream was read and checked against SHA-256
`297c1ee2190f331f5ca80c4906d296faabff8e7460b5cb9466ad3198727a6b43`
before and after inspection. The launch and collection result were also bound,
and all 2,120 original launch source bindings were checked before and after.
No frozen runtime source was changed.

At frames 859–862, the controller requested right turns and admitted measured
bridges numbered 7–10. At frame 863 it requested zero command. Both cameras
still had qualified incremental motion witnesses: primary 83 inliers and
auxiliary 21, each covering eight reference and current image grid cells.
Neither camera had a qualified retained-anchor measurement. All eight retained
references (850, 849, 848, 847, 846, 845, 844, 840) failed their original matcher.
The current pose was consequently unavailable. No goal arrival or round trip
was recorded.

This differs from the earlier failure at 859: recovering one missing increment
does not ensure that the observer can reconnect to an anchor before its bridge
allowance expires. The primary direct-flow pair at 863 passed registration, but
could not supply the missing anchor measurement. Increasing the allowance would
change the continuity contract; these diagnostics did not do that.

A second bounded probe reconstructed the public RGB-D packets for all eight
anchors and frame 863. It bound 41 input files before and after, reproduced all
16 original insufficient-match failures, and applied the existing direct-flow
association twice per pair with byte-exact results. All 16 pairs yielded zero
usable depth correspondences. Fourteen had zero forward/backward-consistent
tracks; the two remaining pairs each had one such track, rejected by the
photometric check. No rigid registration or pose admission was attempted by
this pair probe. The native audit is separate and was not claimed complete.

The existing same-pixel, short-interval association is therefore insufficient
for these retained references. The next investigation is an explicitly separate
association implementation that handles larger view changes, for example
sensor-rotation-guided search with appropriate image-patch geometry. That is a
hypothesis, not a demonstrated fix. Any candidate must retain the original
rigid-fit, gyro, anchor/increment conflict and bridge checks; pass bounded pair
tests and a complete observer/controller history replay; and then be evaluated
in a fresh prospective physics run. No current runtime or queued experiment
has adopted such a change.

Artifacts:

- `docs/go2_direct_flow_bridge_exhaustion_diagnosis_2026-09-11.json`, SHA-256
  `a3d654262ee2f54e9c8a371bbf03b0487545b872762421e834d3d83228bf54d1`.
- `docs/go2_direct_flow_retained_anchor_pair_probe_2026-09-11.json`, SHA-256
  `0f552d0a06450cddb772f7dfcb06a0eab90a78745751c914f9230cc5165b643d`.
- Reproduction sources are the corresponding `diagnose_go2_direct_flow_bridge_exhaustion_v1.py`
  and `probe_go2_direct_flow_retained_anchor_pairs_v1.py` in `scripts/`.

Both scripts completed with exit code zero. The pair probe used one CPU worker,
one OpenCV thread and no model or simulator; its measured elapsed time was
1.59 seconds. Its preflight observed 7% CPU use, about 68.6 GiB available RAM,
about 564 GiB artifact free space and 19.8 GiB workspace free space.
