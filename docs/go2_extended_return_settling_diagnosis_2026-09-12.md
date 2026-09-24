# Extended-return simulation: arrival timing failure

The 8,000-step-budget development run physically returned home on reused
maze 02, with no recorded contacts and verified route retracing, but did not
achieve the required round trip. The return arrival was declared before its
entire one-second window met the physical speed criterion. Preserve this
negative outcome; later quiet samples do not change the declared arrival.

Evidence root:
`/home/andrewknowles/RecoveryStorage/LeWMQuad-v3/.generated/navigation_development_artifacts_v1/go2_extended_return_budget_maze02_v1_attempt_001`.
Case: `no_rgb_direct_extended_return_budget_maze_02`.

- Audit SHA-256: `043154f80da4f3bd02bbe9828ddabc4c7f1a108380b64d788df76dc0706948c1`.
- Readout SHA-256: `95ffaa095fe5d5062dd7404c21dcc744a961df709d58c1ec8b931720104b70fd`.
- Collection completed in 4,202.930293 seconds (70.05 minutes), with 4,749
  observations and 238,150 physical samples. The subsequent paired-prefix
  comparison was interrupted after the complete raw audit. No overall
  pipeline-completion or completed paired-prefix comparison is claimed.

## Physical finding

Outbound arrival at frame 3062 passes: maximum distance 0.04377756 m,
maximum speed 0.04416197 m/s over the required second. Return arrival at
frame 4738 fails: maximum distance 0.02556861 m, maximum speed
0.06086774 m/s, exceeding the unchanged 0.05 m/s limit.

The last movement interval ends at frame 4728 (left arc request
`[0.16, 0, 0.45]`). The next interval uses a zero request but still peaks at
0.06086774 m/s. Its physical endpoint-displacement speed is only
0.04639376 m/s. Thus even an accurate 10 Hz displacement measurement can
miss the instantaneous deceleration peak. The remaining arrival intervals
are physically quiet. The separate terminal stop passes; route retracing,
raw sensor reconstruction, model/command replay and strict visibility pass.

The diagnostic reads `physics_trace.npz` fields `base_twist_world`,
`base_pose_world`, and `requested_command`. For observation frame f, its
physical endpoint is `749 + 50*f`; the evaluated arrival window includes
that endpoint and the preceding 500 samples. These arrays are evaluator-only
and are not controller inputs.

## Prospective correction

`lewm/stop_conditioned_settling_development.py` adds a mission and controller
that require a preceding visually quiet interval under an actual zero
request before counting the existing ten-interval dwell. A movement request
or renewed measured motion invalidates that boundary. The same rule applies
to both outbound and return arrivals. Physical speed/radius criteria, budget,
model, actions, perception and persistent memory remain as before.

The four focused tests in
`lewm/tests/test_stop_conditioned_settling_development.py` pass (1.79 s).
They cover low-motion movement followed by stopping, interruption of dwell,
budget/failure termination, and controller integration. The original running
experiment's source files and artifacts were not changed.

A mission-only replay of the admitted visual positions and actual previous
requests completed in 47.50 s. Goal, phase, hold and termination decisions
matched through frame 4737; the first dwell-counter difference was frame
4729 (original 1, candidate 0). At frame 4738 the candidate correctly withheld
the original return-arrival declaration and remained nonterminal. The replay
stopped at that first behavioral difference and consumed no subsequent
observations. It did not replay the full controller or establish a prospective
physical result. Outbound arrival timing was unchanged in this diagnostic.

Fresh runner: `scripts/run_go2_stop_conditioned_settling_maze02_v1.py`.
It uses the existing resource-guarded collector and complete raw audit, with
the new controller installed in both. Two focused integration tests pass
(2.16 s). The runner preserves an exclusive fresh output, records the original
model/environment and source identities, and runs the raw audit once; it does
not reconstruct the predecessor's full prefix again. It requires the existing
predecessor owner to end and verifies the exact completed raw audit/readout
above before launch. The interruption is recorded in
`docs/go2_extended_return_post_audit_comparison_interruption_2026-09-12.json`.
The comparison had consumed about 25 minutes while reading only about 27% of
its compressed input, with a second pass scheduled afterward. Removing those
additional comparisons follows the user's instruction to minimize development
checks; the physical failure, sensor audit and controller replay are retained.
CPU/RAM/GPU/disk and competing processes are recorded before simulation;
one CPU scene preserves the original numerical settings.

```sh
PYTHONDONTWRITEBYTECODE=1 PYTHONHASHSEED=0 PYTHONPATH=.:lewm_genesis:lewm_worlds OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 OPENCV_OPENCL_RUNTIME=disabled .generated/venvs/genesis_rocm_0_4_6_v1/bin/python -B scripts/run_go2_stop_conditioned_settling_maze02_v1.py
```

This observation rule still cannot bound speed between frames. It must be
tested prospectively with the unchanged physical evaluator. A recorded-input
mission diagnostic may locate its first behavioral change, but cannot prove
the new physical trajectory or qualify navigation. Next, run a fresh case
after the interrupted owners end, preserving this failure. Avoid repeated
whole-history prefix reconstruction in that development iteration.

Independent layouts, matched planner/reactive/non-predictive and memory
comparisons, continuous 100 ms execution and bounded hardware evidence remain
outstanding. This run uses paused simulation during computation and establishes
no real-time or hardware qualification.
