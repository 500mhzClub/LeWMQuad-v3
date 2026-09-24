# Receipt-copy selector integration and paired replay preparation

The prior controller timing diagnostics are complete and were not rerun. The
960-observation phase study identified geometry, map and selection costs as the
main work, with median model inference around 7.3 ms. The current measured-floor
controller already inherits the frame cache. The previously tested receipt
copier remained unintegrated, so this change targets that remaining copy work.

Implemented a separate selector that forks 15 original functions with identical
code objects, closures and defaults. The only changed globals are the receipt
copier and explicit references to other declared forks. A mirrored selector
hierarchy preserves the order of original super() calls. Original module globals,
functions, live controllers and frozen source files are untouched. The candidate
controller replaces only its newly constructed selector and retains the same
residual owner, mission, observer, map, state and complete decision schema.

Tests: 14 selector isolation/copy-semantics tests passed in 1.89 seconds
(59683 exit 0), including the nine previously established copier tests. Eleven
benchmark tests passed in 2.11 seconds (60233 exit 0). They check alternating
order, full decision equality beyond commands, public-input mutation rejection,
timing boundaries, terminal/warmup separation, distinct models and immediate
termination before consuming an observation after a changed decision.

Submitted session 78762, PID 2491693, for all 514 original learned maze-2
observations. Initial live check: 26.65 CPU seconds, RSS 1,330,696,192 bytes;
launch/result/failure files still absent during input authentication. Preserve
this handle and its five new sources. This is an executed submission, not yet
a completed equivalence or performance result.

The benchmark uses two independently initialized copies of the same assigned
corrected model and separate controller histories. At each observation both must
exactly reproduce the saved original decision. The original runs first on even
frames and the candidate first on odd frames. Only controller.observe is timed;
sensor acquisition, packet reconstruction, hashing and receipt I/O are excluded.
Warmup and terminal observations are excluded from active timing aggregates and
reported separately. The eventual result includes both order subgroups. A
replay difference does not establish complete-loop speedup or real-time operation.

Before submission: 16 physical/32 logical CPUs, all 32 in affinity, CPU 3.5%,
74,615,521,280 bytes available RAM, 80,924,983,296 artifact free bytes and
21,359,390,720 workspace free bytes. GPU loads 0% and 1%; discrete VRAM used
1,398,722,560 of 34,208,743,424 bytes. The residual worker was the sole native
scene, with RSS 9,113,477,120 bytes. One CPU replay process with a 16 GiB memory
allowance fits the measured headroom alongside that scene. Resource admission
is refreshed after input verification and monitored during replay; these are
capacity checks, not OS quotas.

Source SHA-256 identities:

- `lewm/receipt_copied_selector_development.py`:
  9c3e06032df9f82fdf7ba9d7cc05477f9c324b3a5262e286fa2375edddf9c075
- `scripts/benchmark_go2_receipt_copied_controller_v1.py`:
  e7569060a4be7fbc338931ec0b40b1143334309eadf6ecc9ad6a026b69a3003b
- `lewm/tests/test_receipt_copied_selector_development.py`:
  47abf0392e9eb097d5524950a8d6797d4bd3b2bd406c217a87bea1db1f565115
- `lewm/tests/test_receipt_copied_controller_benchmark_development.py`:
  2aceabeccf12a7cdb622b26e94a711fc1522561d529c871767adcfa62b4e2193
- `docs/go2_receipt_copied_controller_benchmark_v1_2026-09-09.md`:
  1891da09598d7d732dd817ee0d39b15e8dd5691423bf10cf7c08016fa1f9379d

No native implementation is changed or promoted. Await full equivalence,
measured paired timing and final artifact verification before deciding whether
this implementation is useful. Preserve a mismatch or negligible/negative speed
difference. The navigation goal remains active and unachieved.

## Launch admission complete; replay active

Same session78762 passed launch admission, SHA-256
96f6a1b90454689d0fe8b6f602ba178535be58054d2401e45553f586e8ea569e,
1,671 frozen sources. Launch resources:73,319,276,544 bytes available RAM,
80,380,456,960 artifact free bytes, CPU3.6%, GPUs idle, all32 affinity. The
initial live replay check counted216 completed pairs, each with complete original
and candidate decision equality. PID2491693 was running at490.93 CPU seconds,
RSS2,794,754,048bytes. This is partial equivalence evidence only; keep the same
run through all514 observations and final source/artifact checks. No timing
aggregate or speedup conclusion is selected from the partial stream.
