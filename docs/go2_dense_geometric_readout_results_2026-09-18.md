# Frozen-feature geometric readout diagnostic

One fixed, untrained geometric decoder was evaluated on the completed stall
branches. Mutual nearest feature matches use the 24x32 V-JEPA patch grid. Current
delivered depth supplies 3-D points; RANSAC PnP estimates the future camera pose,
which is converted through the fixed camera mount to body motion. No future
depth, native pose, commanded motion or fitted physical head enters the decoder.
Native poses are used only as evaluation targets. The encoder and predictors
were unchanged. No navigation run used this decoder.

The fixed synthetic planar-wall check failed its 4-cm / 0.025-radian accuracy
limits: quantized perfect patch identities still produced 11.3 cm spurious XY
translation for one pure turn. Both signs and identity motion were checked;
the negative result is retained in
`go2_dense_geometric_readout_synthetic_check_2026-09-18.json`.

On the actual exposed branch scene, all twelve estimates were finite and passed
the decoder's match/inlier requirements. Those requirements do not certify
accuracy. Metrics use three actions at 300/500/700/800 ms; the pre-branch 300-ms
case repeats across actions, so these are not twelve independent trials.

| Features | Original MLP XY RMSE | Original MLP yaw RMSE | Geometric XY RMSE | Geometric yaw RMSE |
|---|---:|---:|---:|---:|
| Actual future (offline oracle) | 25.11 mm | 5.70° | 13.90 mm | 0.41° |
| Action-conditioned prediction | 30.82 mm | 6.06° | 79.53 mm | 3.11° |
| Action-blind prediction | 27.00 mm | 6.17° | 60.55 mm | 6.15° |

For actual left and right turns over 300–700 ms, geometric oracle yaw was
+9.01° / -9.03°, compared with physical +9.17° / -9.24°. On predicted features,
the corresponding estimates were +14.46° / -15.53°, with spurious XY changes
of roughly 118 mm / 150 mm. The candidate turn signs improve, but the physical
forecasts remain inaccurate. This decoder is **not selected for navigation**.
The eight-second probe retained complete per-case results under
`.generated/navigation_development_artifacts_v1/go2_dense_stall_turn_branches_v1_attempt_001/geometric_readout_v1/`.

The oracle result establishes useful geometric information in the actual frozen
features at this context. The worse predicted-feature result shows that merely
replacing the readout does not solve the entire forecasting problem.

An additional output-subspace diagnostic examined the predictor's 384-to-1024
linear output and subsequent layer normalisation. Its centered affine span has
rank 385. Orthogonal target-projection error at the stalled context was only
0.019–0.023 MSE, much smaller than the observed roughly 0.43–0.56 prediction
errors. Three illustrative training frames gave 0.009–0.010. This does not
support blaming output dimensionality alone, or establish that a wider head
will solve transfer. It ignores the trunk's nonlinear reachability constraints.
Results are in `go2_dense_predictor_output_subspace_probe_2026-09-18.json`.

Next direction: improve training-view coverage while retaining the encoder and
the completed predictors as fixed baselines. A compact collection on existing
training geometries can add full-heading views and balanced turn/pulse motion;
the exposed failure and the four prospective mazes must remain outside fitting.
A matched old-data continuation accompanies mixed-data readout training, so extra
optimization is separated from added visual coverage. The eight-recording
collection has completed with 2,832 windows and no contacts; matched readout
training is now running. See `go2_full_heading_readout_experiment_2026-09-18.md`
for the fixed treatments, artifacts and live-process identity.
