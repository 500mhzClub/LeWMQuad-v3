# Translating view-recovery prefix: first changed command 595

The completed prefix is `go2_view_reentry_maze_prefix_v1_attempt_001`, result
SHA-256 `6d550308c908eca596149df684f639a032c1cef706a36e19aa31cff7a5cfd3f4`.
Launch SHA-256 is
`d99dc111015ffd3ae1859b0d6cf554349e5667cbcd854ea5e909b3525ea38bd6`,
binding 1,447 sources. Decision stream SHA-256 is
`17c45e505d372bcca54e47541b673dee9604649798b8bbd5b6faa8497b7f5e64`.
Session 65832 completed; post-launch wall time was 387.114 s.

The fresh controller replayed 596 observations from the completed
executed-waypoint native attempt. Causal observation/map/mission/residual evidence,
raw forecasts, original surface/nominal-path/phase receipts and all other
pre-intervention decision fields matched exactly. View selections reconstructed
the entire frozen predecessor transformation before checking the new transform.
The assigned corrected JEPA state remained
`4f796afdf32c2c6dbcd1d5981834a7b17be86e2efdafd9427b2d80240def0ca6`,
with unchanged parameters and absent gradients.

The first changed request was tick 595: right arc `[0.16, 0.0, -0.45]` instead of
`[0.0, 0.0, 0.0]`. No terminal or sensor/model failure occurred. The controller
was acquiring a view, with original phase allowance hold/left-turn/right-turn.
Current observed nominal clearance was 0.449071776928 m to cell [13,12]. All six
surface checks passed. Hold, left arc and both turns worsened predicted clearance.
Forward and right arc were nonworsening across all eight raw predicted segments
and had positive first-endpoint clearance gain, but were excluded by the old
view phase. The explicit recovery exception admitted both.

Right arc's predicted first-endpoint clearance was 0.450141782426 m, a
1.070005498 mm gain. Its full-plan contact score was 0.0364070985845 and recovery
utility -0.0426185128031 m. Forward's gain was 0.870639909 mm, contact score
0.0596491762258 and utility -0.0707083715620 m. The unchanged recovery utility
therefore selected right arc. The original nominal veto and phase receipt remain
recorded, with zero eligible candidates under the original phase and two under
the explicit recovery allowance. No model-error or physical-clearance guarantee
is established by this millimetre-scale predicted improvement.

This intervention occurs well before the predecessor's terminal stop at 1155.
No observation after command 595 was consumed by the new controller. The old
continuation is not the new policy's executed outcome, and this result does not
prove that the new policy reaches the later stop, restores physical clearance or
improves navigation. A new native trajectory and raw audit remain necessary.

Before launch, available RAM was 82,725,384,192 bytes, artifact free space
51,640,266,752 bytes, CPU 0.4% busy and GPUs idle. After completion, available RAM
was 81,607,897,088 bytes, artifact free space 51,608,977,408 bytes, CPU 0.3% busy
and GPUs idle. Only one CPU replay ran; no scene or model training was launched.
All prefix output bindings were checked after completion. The launched sources
remain frozen. This adds zero native episodes, independent layouts, verified
arrivals or round trips. Reactive baseline is still first in the native queue.
