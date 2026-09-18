# Direct-model maze-2 visibility failure localized

The recorded strict failure at frame 1320 is one sampled primary-camera pixel,
row 308, column 508. Its centre lies 0.00001492891 pixels from the projected
vertical edge of wall `novel_wall_1_0_1`. The ideal centre ray reaches the floor
at optical depth 1.9103261277 m, while native depth is 1.3680409193 m, near the
wall. The 3-by-3 depth neighborhood crosses that same wall/floor discontinuity.
Adjacent frames 1319 and 1321 have no failed sampled rays.

An independent scalar ray calculation reproduces the floor depth and projected
edge distance. At the wall-edge x plane, the centre ray is only
5.36012195e-8 m beyond the nominal wall's y boundary. This is consistent with a
raster boundary discrepancy. Proximity alone does not identify the exact cause;
renderer projection, finite precision and coverage have not been independently
reconstructed. No scene was created or rerendered.

The three original footprint reports were recomputed exactly from bound native
depth buffers, original camera transforms and original scene geometry. At the
failed frame, 4,337 stable-interior rays pass with maximum error 0.0002689585 m;
one of the 189 boundary-ambiguous compared rays fails. This explains why the
separate hard-measurement failed-frame list is empty. Boundary pixels remain
uncertified, and the complete episode's original strict visibility result
remains false. This diagnosis does not repair its zero-traversal navigation
result or qualify its sensing.

The diagnostic authenticated 1,910 source bindings and the explicitly selected
original artifacts before and after reading. It checked the native buffer hash
and reproduced the original frame scores. A separate verification authenticated
those bindings again and checked edge distance and ray-plane geometry using
scalar arithmetic. These are three-frame evaluator checks, not a rerun of the
full original raw sensor/model/command audit.

- Diagnostic: `go2_full_direct_maze02_visibility_diagnosis_2026-09-10.json`,
  SHA-256 `cfa6849af9b037acc8d75f47289fcdf8df7c9cb28cda1a10321a988117cd6f53`.
- Independent verification:
  `go2_full_direct_maze02_visibility_diagnosis_verification_2026-09-10.json`,
  SHA-256 `71d1a33b2258795dfff30e60165a89cea0e32a3c5650b8cac0851cf86035aa5b`.
- Source: `scripts/diagnose_go2_full_direct_maze02_visibility_v1.py`.

The first source check invocation failed before diagnostic output because the
binding verifier was passed a wrapped map. That call was corrected to use its
actual flat-map API; the completed invocation and independent verification
exited zero. No original evidence or live source was modified.

Retain the existing strict failure and coverage accounting. Any prospective
sensor or evaluator change needs separate validation; removing this pixel or
relaxing the threshold would not establish accurate sensing or useful navigation.
The queued policy experiments and full-history performance diagnosis remain the
next execution priorities.
