# Body-projection comparison verified; external sampling tool prepared

The body-projection replay and its automatic completion watch both ended
successfully. Tool sessions 48899 and 80027 exited zero; owners 2876376 and
2876551 have ended. Do not repeat the completed verifier.

- Replay root: `go2_body_projected_tiled_late_history_v1_attempt_001`.
- Launch SHA-256: `69dd113461f8301d6d47daf9f0174441aed8f5047a27dd68de1c5103f18c652f`.
- Result SHA-256: `548c8afe30d87c5f2d503c820a77a03c75d7c764dd6d045529d1a258fad94eea`.
- Execution SHA-256: `39755d9dbec68b6fd5525dd9389c741a22d4e7ae0e30626a1fbe7fe91092e351`.
- Completion:
  `docs/go2_body_projected_tiled_controller_completion_verification_2026-09-11.json`,
  SHA-256 `47c6e00d30f9b7e8c31b19dbeed511699d7565145e2b34273fe56e3c96eb25ab`,
  2,376 source bindings.
- Completion-watch result:
  `e6690cad09226250c61ef70d706ab963a0b943de31bf8c0643ed2ac467fd16c9`.

All 1,428 original observations/decisions, 1,425 model forecasts and seven
original retained-state witnesses matched. Complete timing populations and the
original raw/model inputs were reauthenticated. The original strict sensing
failure at frame 1173 remains; there is no navigation qualification.

| Window | Tiled baseline total seconds | Body-projection candidate total seconds |
| --- | ---: | ---: |
| All 1,425 navigation observations | 823.230965 | 805.987553 |
| Early ten observations | 4.515806 | 4.288566 |
| Repeated-hold ten observations | 6.214966 | 6.002795 |
| Late ten observations | 7.014708 | 7.553238 |

The total reduction is 2.094601940729812%. Overall medians are 546.840717 ms
and 532.844614 ms. Late-window medians are 672.453056 ms and 758.950261 ms.
Every navigation observation in both arms exceeded 100 ms. Preserve the late
regression and shared-host scope. No native controller adopted this optimization.
Neither these percentages nor earlier component savings may be added together.

## Internal sampler: retained calibration and limitation

`lewm/wall_stack_sampler_development.py` provides bounded wall-time Python-stack
sampling without profile/trace hooks or frame-local/source-file reads. Seventeen
tests passed in 0.17 seconds. It preserves original computation exceptions,
rejects truncated/capacity-exhausted success and joins its sampling thread.

The fixed synthetic calibration completed in session 18053, exit zero:
`docs/go2_wall_stack_sampler_calibration_2026-09-11.json`, SHA-256
`3bd4a2717980b018becb7475ddb8e433e24e18ea775d8d21fb99706fccf55b1e`.
It preserves two warm-up pairs and twenty measured pairs for each workload,
all sample traces, exact return-value equality and unchanged synthetic-array
bytes. Its bindings cover four explicit local files, not a recursive closure.

| Workload | Total elapsed overhead | Median actual inter-sample gap |
| --- | ---: | ---: |
| Python integer loop | 0.609541% | 15.134256 ms |
| NumPy sine/sum operations | 0.275377% | 10.089695 ms |
| Waiting | 0.377114% | 10.086955 ms |

The nominal interval was 10 ms. The observed difference in actual sampling gaps
prevents treating unweighted sample frequencies as unbiased cross-workload
CPU-cost estimates. Keep this calibration and limitation. No current controller
or native process was instrumented. Small synthetic overhead does not establish
controller overhead or remove GIL scheduling bias.

## Pinned external profiler and actual child-process smoke check

No installed `py-spy` or `perf` executable was found. A py-spy 0.4.2 binary was
installed separately under `.generated/tools/go2_py_spy_0_4_2_v1/`; the existing
runtime environment and system ptrace settings were unchanged.

The [PyPI release](https://pypi.org/project/py-spy/0.4.2/) supplied the pinned
x86-64 wheel and published SHA-256. Its hash was verified before extracting only
the executable. The [upstream documentation](https://github.com/benfred/py-spy)
describes profiling a child launched by the profiler and explains the idle/GIL
filters. The actual local child-process check below establishes availability on
this host; documentation alone is not execution evidence.

- Wheel: `py_spy-0.4.2-py2.py3-none-manylinux_2_5_x86_64.manylinux1_x86_64.whl`.
- Wheel SHA-256: `aeb0323409199c785f730645e9f4bb7a7b9ca2c481f2c331a55642b5d13fa52f`.
- Executable SHA-256: `9b4d1f39b2a47ae44f4c6a46f615dcc0287d7755beba5065f32391951e07d594`.
- Installation metadata and original PyPI JSON are retained beside the binary.
- Child source: `scripts/py_spy_workload_smoke_child_v1.py`.
- Smoke root: `.generated/tools/go2_py_spy_0_4_2_v1/smoke_v1`.
- Smoke result SHA-256: `6e4364e4361a99f6b7ada33389bb0dd8d42a6e03d360e675b637c60c44b5d3fc`.
- Tool session 28521 exited zero. Both original and externally profiled child
  commands exited zero and returned exactly the same workload values.

The smoke used `record --format speedscope --rate 100 --idle --threads
--full-filenames -- <python> -B <child>`. No GIL-only filter or nonblocking
sampling was enabled. It observed 97 Python-work, 103 array-work and 104 waiting
samples, approximately 107, 104 and 104 samples per measured child-workload
second. These short-run counts demonstrate coverage, not unbiased attribution,
production overhead or native-stack coverage. All commands, stdout, stderr,
profile data, source identities and artifact hashes are retained. No running
navigation or controller comparison was attached to the profiler.

## Next bounded diagnostic

The full CPU replay slot is now free. Prepare a separate external-sampling
diagnostic for the completed body-projection controller, retaining the original
full 1,428-observation history and actual input/model/decision/state checks.
Use explicitly identified controller windows so initialization and repeated
input admission cannot be mistaken for controller cost. The profiler must launch
its own new child; bind the profiler binary, child invocation and both process
identities. Require both to end before reading the completed profile. Define
profile completeness and stack filtering prospectively, preserve failures and
do not use profiled elapsed times as a new unprofiled speed comparison.

The extended-budget native worker 2867880 remains active; last checked timing
row was tick 3215, beyond the original budget boundary. This does not establish
prefix equality, outbound arrival or return success. Await its original terminal
audits and preserve the downstream sustained-turn, contact/flow and chained-anchor
queue. The full unseen-maze navigation goal remains active.
