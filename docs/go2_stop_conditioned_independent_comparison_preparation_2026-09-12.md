# Updated independent-maze comparison preparation

The updated controller can now use the fixed eight-layout independent
development inventory with the same 8,000-step sensing/physics pipeline and
physical evaluator. No independent simulation has run with this integration.

`scripts/stop_conditioned_independent_maze_pipeline_development.py` composes
the existing independent scene initializer with the extended RGB-D session.
Both collection and audit use the independent specification and coordinate-only
public mission. The controller is not given the scene graph. The evaluator
retains the original physical arrival, speed, contact and backtracking criteria.

`lewm/stop_conditioned_comparators_development.py` and
`scripts/stop_conditioned_comparator_pipeline_development.py` apply the same
stop-conditioned dwell to all four prepared modes:

- `frozen_reference`: assigned learned action-conditioned forecasts.
- `nominal`: requested-twist forecasts with the original scoring, feasibility
  and observed residual correction; this remains a predictive baseline.
- `reactive`: no high-level model or candidate future evaluation; a whole-method
  comparison, not an isolated change to prediction ranking.
- `current_planning`: current-observation routing-map ablation, retaining visual
  tracking, contact history and residuals; not a fully memoryless controller.

The four comparator tests pass (1.80 s). Eight independent-integration tests
pass (2.51 s), covering actual constructor dispatch up to the scene builder,
extended acquisition, both scene bindings for each arm, and long-trace negative
evaluation. These are component evidence, not end-to-end navigation evidence.

The corrected maze-02 trial is running as PID 3130098, process creation time
1789252811.73, under launch SHA-256
`bbc6de995fd879806f3083bbd44da88e9b37b9bc4c7753ec40a5307872070802`.
Do not duplicate it. Its runner is
`scripts/run_go2_stop_conditioned_settling_maze02_v1.py`; monitor the exact owner
and launch/result/failure files before taking a new execution action.
The previous queue (PID 3128161) and the original post-audit comparison were
interrupted to remove hours of additional comparison work. The original
complete sensor/controller audit and negative physical outcome are preserved;
its overall pipeline did not complete. See the interruption witness in
`docs/go2_extended_return_post_audit_comparison_interruption_2026-09-12.json`.

Next after that trial: freeze a prospective independent comparison using the
actual corrected model assignments, with matched training seeds, layout order,
appearance, budget and sensors. Use JEPA versus supervised prediction to assess
training, nominal/reactive controls to assess online prediction, and
current-planning to assess accumulated routing memory. Preserve all failures;
do not select replacement layouts from outcomes. Assess available resources
before choosing the batch size. The old 32-case, single-seed, 4,000-step roster
does not describe this updated experiment and must not be launched unchanged.

Single-case runner: `scripts/run_go2_stop_conditioned_independent_case_v1.py`.
The first independent development case is fixed as layout 0, frozen-reference
planning, `seed_2026091001_full_jepa` (the original primary candidate). The
runner accepts only registered model assignments and the eight fixed layouts;
it records the loaded corrected model identity before any scene execution.
It reuses the completed model admission rather than rerunning training-ledger
or coefficient reconstruction. This first case alone cannot establish
reliability, comparative advantage or training-seed repeatability.

Eight focused single-case runner tests pass (2.36 s). The fixed first case is
queued as PID 3131637, creation time 1789253560.06, waiting for exact stopping
trial owner PID 3130098 / creation time 1789252811.73 to end with a completed
result. It does not require a positive scientific outcome and performs no
automatic retry. If the preceding process ends with an operational failure,
the queue stops. No independent scene has started at the time of queuing.

```sh
PYTHONDONTWRITEBYTECODE=1 PYTHONHASHSEED=0 PYTHONPATH=.:lewm_genesis:lewm_worlds OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 OPENCV_OPENCL_RUNTIME=disabled .generated/venvs/genesis_rocm_0_4_6_v1/bin/python -B scripts/run_go2_stop_conditioned_independent_case_v1.py --layout 0 --mode frozen_reference --model seed_2026091001_full_jepa
```

Real-time execution and hardware validation remain outstanding. Current
simulation pauses physics during controller computation.
