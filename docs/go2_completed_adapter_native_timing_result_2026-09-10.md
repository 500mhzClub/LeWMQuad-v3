# Completed adapter episodes: full-history timing diagnosis

The dominant measured cost is controller work, and it grows substantially
within each completed episode. The earlier 405-observation optimization replay
does not establish late-episode timing. This goal turn made progress by reading
all 4,543 saved timing rows from the completed JEPA and supervised cases,
authenticating their original selected artifacts and reproducing both original
complete timing readouts exactly. It made no new native or model execution.

| Original development case | First 100 navigation decisions: controller median | Last 100: controller median | All navigation: acquisition median | Controller share of total recorded iteration time |
| --- | ---: | ---: | ---: | ---: |
| JEPA | 776.83 ms | 2299.98 ms | 237.93 ms | 85.01% |
| Supervised | 1109.50 ms | 3671.26 ms | 203.54 ms | 90.78% |

These are separate original policy histories on the shared simulation host.
They do not establish a causal speed difference between training objectives.
The table excludes three warmups and eleven terminal observations per case;
all 1,515 JEPA and 3,000 supervised nonterminal navigation decisions remain.
Warmup/terminal timings are preserved separately in the complete result.

Within the same JEPA episode used by the earlier optimization replay, the first
402 navigation decisions had a controller median of 784.20 ms. The following
1,113 had a median of 1919.60 ms. The original full navigation median was
1760.11 ms. The earlier optimized candidate's approximately 664 ms median and
13.5% paired total reduction applied only to that first 402-decision replay;
neither number is a whole-episode result. No optimized late-episode timing has
been established here.

Acquisition remained comparatively stable. Even that component alone exceeded
the 100 ms interval in every navigation observation in both original cases.
The simulated primary/auxiliary render, file and packet pipeline is not a
physical-camera latency measurement. The existing renderer-context query cost
was only about 0.31 ms per paired query; removing those witnesses would not
address the measured dominant cost. Captures and iteration intervals can be
nested or straddle acquisition boundaries, so the component medians must not
be summed to invent a new total.

The already completed frozen-footprint cProfile provides a second useful clue.
Its exact model `AllPhaseTranslationBiasModel.forward` ran ten times in each
window, totaling 0.074573759 s in early navigation and 0.073294760 s during
repeated holds: approximately 7.46 and 7.33 ms per call under profiling.
These cumulative model times must not be added to their nested forward calls.
The original profile files were reauthenticated:

- `early_navigation.prof`: `e06357c5825b1b0c8734d419b3740877868edc74edb20bf11ceb00e048432ae3`
- `repeated_hold.prof`: `32bd4983ea64a1262e8f7517c645ae10c327e7be07f61a1ffa42076d5de36651`

Both belong to profile result
`c636eb55c13f02624b73680295ab3f70d7faac00680d7e66dd820b870cfb9866`.
That profile identifies retained-floor coverage and receipt copying/construction
as substantial work. Neural inference acceleration is therefore not the primary
next target. The full-history timing trend alone does not prove which retained
data structure causes the growth; a late-history profile is needed before
choosing the next optimization.

The analyzer is `scripts/analyze_go2_completed_adapter_native_timing_v1.py`.
Execution session 80959 exited 0 in 45.93 seconds. It bound 2,076 sources and
read every original compressed decision row plus the matched capture/write
timing records. Selected artifact hashes were checked before and after reading.
No complete raw sensor/controller audit was repeated. The original JEPA strict
visibility failure and both negative navigation outcomes are preserved.

Result: `docs/go2_completed_adapter_native_timing_2026-09-10.json`, SHA-256
`6d5e21e96d916cc3d0ab7bee540121b25e06f230e4b630a307ae665bca4d2d82`.
Figure: `docs/go2_completed_adapter_native_timing_2026-09-10.png`.
The plot uses medians of successive groups of up to 100 navigation decisions;
the final JEPA group contains 15 observations. It is not a fitted latency model.

Independent numerical verification session 31786 exited 0. It recomputed 464
statistic groups with standard-library median, explicit linear p95 interpolation
and `math.fsum`, including all declared windows and plot bins. P95 differences
were zero; the largest total-time difference was 9.32e-10 ms. It also
reauthenticated the selected original artifacts and source bindings.
Verification: `docs/go2_completed_adapter_native_timing_verification_2026-09-10.json`,
SHA-256 `4273c250cc3df21835dfae4dec15318cdf6a65c716a7244489265fef30b0913e`.

Next, prepare a fixed late-history controller profile on existing development
observations, rebuilding the complete causal history and checking original
decisions and public-input integrity. Preserve the known sensing failure and
make no qualified-visibility or navigation claim from that profiling tape.
Measure the dominant late-history work before another short-prefix optimization.
Keep the original six-case/frontier/hold/contact native queue unchanged. The
goal remains active: 39 completed audited development episodes, zero verified
round trips, and unresolved independent navigation, backtracking, attribution,
realistic timing and hardware validation.
