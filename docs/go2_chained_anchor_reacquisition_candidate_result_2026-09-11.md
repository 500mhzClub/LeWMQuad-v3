# Chained image correspondences: candidate anchor recovery before bridge exhaustion

Implemented `lewm/chained_corner_flow_association_development.py`. It follows
only original reference corners through consecutive measured images, using the
existing short-interval forward/backward, photometric and depth gates at each
link. Lost tracks cannot be reseeded. Endpoint 3D points are lifted directly from
the original and final depth frames; pose increments are never composed. The
history is bounded to 32 intervals of exactly 100 ms. This is an association
component, not an integrated or adopted observer/controller.

The 12 focused tests passed in 0.28 seconds, covering equality with the existing
one-interval matcher, known 96-pixel displacement over 12 links, exact original
endpoint lifting, repeatability, unchanged inputs, occluded and invalid-depth
intermediate frames, invalid clocks/populations/images, and duplicate features.
Command:

```sh
PYTHONDONTWRITEBYTECODE=1 PYTHONHASHSEED=0 PYTHONPATH=.:lewm_genesis:lewm_worlds OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 OPENCV_OPENCL_RUNTIME=disabled .generated/venvs/genesis_rocm_0_4_6_v1/bin/python -B -m pytest -q lewm/tests/test_chained_corner_flow_association_development.py
```

The first raw probe tested all eight retained anchors in both cameras against
the original terminal frame 863. All 16 fits failed. The auxiliary camera
retained 1–10 endpoint correspondences, below the original 12-match minimum;
the primary camera retained zero. This negative result remains preserved.

A separate follow-up tested the same component and thresholds at frame 853.
The complete 874-row closed stream verified that 853 was the first measured
bridge, before the ten-frame allowance expired. Of the same 16 reference/camera
pairs, exactly one passed endpoint registration: auxiliary reference 850.

| Measurement | Candidate at frame 853 |
| --- | ---: |
| Endpoint depth pairs | 42 |
| Rigid-fit inliers | 30 |
| Inlier fraction | 0.7143 |
| Reference/current grid cells | 8 / 6 |
| Fit RMS | 0.699 mm |
| Gyro disagreement | 0.000538 rad |

These use the original rigid-fit and gyro rules, including their original
minimum support and rejection thresholds. The relative gyro was reconstructed
from the actual public fast/slow sensor packets over the bounded interval.
No native robot pose was supplied to the pair fit.

A further check composed that endpoint fit with the *recorded visual* pose of
retained reference 850. Its distance from the previous frame's recorded visual
pose was 1.228 mm, with 0.04796 rad rotation, inside the original increment
envelopes. It agreed with the original qualified auxiliary increment at frame
853 to 0.136 mm and 0.000252 rad, inside the original disagreement gates. This
uses original recorded public observer evidence; it is not an independently
replayed observer state or a ground-truth accuracy result.

All probes checked their source and input hashes before and after execution,
and compared repeated association arrays byte-for-byte. No model, controller,
physics attempt, bridge allowance, or registration threshold was changed.

The next implementation is a separate full-observer candidate that attempts
chained retained-anchor measurements when the original observer would start a
measured bridge. It must preserve original qualified-measurement conflicts,
reference/promotion rules, gyro history and terminal behavior, and account for
bounded image storage and processing cost. A full history replay must verify
the resulting state transitions before any new controller/physics experiment.
The saved pair result alone does not prove that changing the observer at 853
will avoid later failure or improve navigation.

Artifacts and SHA-256:

- Terminal-frame negative probe:
  `docs/go2_chained_retained_anchor_pair_probe_2026-09-11.json`
  `bc8ec3fc6dc942ed1f9eff11449e5ae716736ceeb833e71acd7fafd296a77508`.
- First-bridge all-pair probe:
  `docs/go2_chained_first_bridge_anchor_pair_probe_2026-09-11.json`
  `7e2dd4bf40d066d558f61ec1b9ecf901230a82aa62886901cf2537ee4ef90660`.
- Recorded-pose envelope/conflict check:
  `docs/go2_chained_first_bridge_pose_candidate_verification_2026-09-11.json`
  `d1f581b346815994e35160270f48ee4cfc63f6bc185155b6c7cb0fd586004a03`.

The three corresponding reproduction scripts are in `scripts/`. All completed
with exit code zero. These records bind the new component and tests; preserve
them as executed source when developing the observer integration.
