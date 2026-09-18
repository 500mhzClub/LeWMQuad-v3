# Dual-camera native episode: settled arrival repeated, return still fails

Eleventh native episode and readout are complete. The independently audited
physical trajectory again settles at the outbound goal, then executes return
turns until floor-plane extent admission fails at1904. No return edge is crossed
and no round trip is verified. Only reused development maze0 was exercised.
The outbound prefix repeats the previous episode exactly, so this is not new
independent-layout evidence or a matched contribution result.

Native root go2_dual_camera_settled_maze_pilot_v1_attempt_001:

- launch afb8485c377a12989004cdc7827476948d5308b3b89387a7327b385c2739cba6;
  1616 frozen source bindings;
- result44710966178a57b31f7da3bec10ad4f21710bcff3701b0a1038750f1ef6d747c;
- audit fd16aee42c6ab0b2439e427f25fad5aa45dce4e5678046efdc409cf491057020;
- prefix comparison2664beb2f117d5e5f66532b983f82d7fc21103edd64a98a2193a882c9f9e03d7;
- collection dc5d54e1372b67ae88c783cfb06034777c3e50eccdba40f2aeb1dc5bb3f4fdc5;
- session67399 CLOSED exit0; wall5138.616231464s.

Readout root go2_dual_camera_settled_maze_readout_v1_attempt_001:

- launch42a54ce75e58d37bea443f26ca87756ba92cb6c2ad5f6d0eb9a2e602ca7fa8cf;
  1618 frozen source bindings;
- result6fbbf9a63ca770317fc929c49796993ff3cf9b268cb4572d2d39a7bfd660d5db;
- session90363 CLOSED exit0.

Collection contains1915paired observations/decisions,1914completed command
intervals and96450physics samples, including ten terminal zero commands.
Physical/acquisition stop fields are null; controller terminal is
SENSOR_OR_MODEL_FAILURE. Raw sensor reconstruction (including added auxiliary
RGB), full fresh model/controller replay, actual command audit and model-state
checks all pass.1873physical/public/prospective decision frames through the
first auxiliary intervention match their bound prefix; raw physical prefix
SHA776e8f9c64c50deecae9aef56dca751134a01d182580eed43aa1d735db11d244.

Native one-second settled arrival at1868 passes: maximum outbound goal distance
0.037619440722m and maximum3D speed0.039015776111m/s. The loop-erased outbound
route has six valid open edges and no invalid crossings. The return stays in
the destination cell. Terminal native quiet and reverse-route completion do
not pass. Strict primary visibility fails only at909; auxiliary visibility
passes on every frame. There are no hard-measurement failed frames. The strict
failure remains, so the experiment supplies no navigation qualification.

Auxiliary visual tracking is selected at1872,1873,1877,1878,1879,1903,1904.
There are1897primary selections.1904still has a valid raw visual pose but no
admitted floor-registered pose; last admitted registered frame is1903.
The generic floor validator error is independently reconstructed as insufficient
combined two-axis extent, not arithmetic moment disagreement. See
docs/go2_dual_camera_floor_extent_diagnosis_result_2026-09-09.md.

Native path length9.027919650m over191.4simulated seconds. Minimum outbound-goal
distance0.021496481m, terminal distance0.021661194m. Across1904admitted registered
poses, mean/max XYZ error is0.004210428/0.008581741m; mean/max XY error is
0.004208595/0.008577553m; mean/max rotation error0.004309359/0.010904357rad.
These are retrospective errors on this executed trajectory, not calibrated
online uncertainty or bounds on unexecuted candidates.

Receipt-inclusive iteration median1263.296772ms, maximum1951.748733ms; all1915
iterations exceed100ms. Physics pauses during computation. The separately
tested map-performance candidate was not used here. No real-time or hardware
qualification follows. Model4f796afdf32c2c6dbcd1d5981834a7b17be86e2efdafd9427b2d80240def0ca6
is unchanged and no new training occurred.

Two completed episodes now independently audit the same settled outbound
arrival on maze0; zero episodes verify a full return. The next transport
candidate must finish its complete causal prefix before a new prospective
episode. Independent layouts, matched reactive/non-predictive comparisons,
JEPA/prediction/memory attribution, timing and deployment-valid validation
remain outstanding. All failures are retained; the full goal remains active.
