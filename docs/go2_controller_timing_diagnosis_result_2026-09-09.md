# Completed controller timing diagnosis

The unchanged confirmed-floor controller reproduced every complete decision
exactly for observations 0–100 of the completed seventh native maze-0 attempt.
The assigned model state remained unchanged. No native scene, new command
outcome, training or controller change occurred in this diagnostic.

Artifact root: `go2_controller_timing_diagnosis_v1_attempt_001` under the external
navigation development artifact base. Result SHA-256:
`b7dfb633bbaa585d2346c87fee9889be2787d8a97cfa01b67c638699bbafafab`.
Launch SHA-256:
`7d4867e6a916179031a4ff004264f72167d07e00c684c323f0f36e77f54580ea`.
1480 sources and the completed native artifacts were verified before and after.
Session49066 completed successfully. Source and protocol remain frozen.

Of these first 101 observations, 95 non-warmup, unprofiled controller calls had
median wall time 798.479455 ms. This is a short beginning-of-trajectory replay,
not the complete native run's timing distribution. The complete predecessor's
1548 recorded observations had median acquisition213.722208 ms and controller
884.892166 ms, with combined median1093.312998 ms. Separate medians need not sum.
Acquisition and physical execution are absent from this controller replay.

cProfile was enabled only at frames20,60,100. Those calls took approximately
1.23–1.28 seconds including instrumentation overhead. Cumulative profiler
times below overlap through call nesting and must not be added together.

| Function/group | Profiled cost per selected step | Interpretation |
| --- | --- | --- |
| `copy.deepcopy` | 0.456–0.465 s cumulative, roughly674k–683k total calls | Repeated nested evidence/selection copies are substantial. |
| `observed_floor_cell_index` | Nine calls, 0.291–0.328 s cumulative | Full measured floor mesh/index computation is repeated. |
| Auxiliary floor map `observe` | 0.633–0.696 s cumulative | Includes several floor computations, insertions and copies above. |
| Measured sample-bound `insert` | Eight calls, 0.179–0.200 s cumulative | Repeated return accounting/indexing also contributes. |

A separate small component measurement reproduced saved selected-reference
visual fits at frames100,600,1500 exactly. Each rigid registration took roughly
10.6–13.1 ms unprofiled; two feature frames took17.3–35.2 ms and matching3.6–11.0 ms.
It did not time all attempted references or the complete observer. Temporary
report `/tmp/go2_visual_fit_timing_20260909_result.json` has SHA-256
`8bc73214ea0e3ed24d299db79493a75f951a8b2cadaa5288e86166b55b48f121`.
The authoritative complete-controller diagnosis above independently establishes
the larger costs; no claim relies on previously lost diagnostic stdout.

Next performance work should identify which repeated floor computations have
exactly identical depth, validity and up-vector inputs, then explicitly share
immutable per-observation results in a separately named implementation. Camera,
timestamp, calibration, up-vector and invalidation distinctions must remain
intact. Review nested-copy ownership before replacing copies with shared data;
aliasing must not alter retained historical evidence. Require complete decision
equality on raw replay before adopting an optimization. No cache, copy change,
numerical change or optimization has been installed in the running native pilot.

Neither these component times nor the profiled times prove achievable full-loop
speed. The 100 ms real-time target, sensing qualifications, independent-layout
arrival/return and matched planner/baseline comparisons remain outstanding.
