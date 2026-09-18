# Chained full-history timing launcher prepared and source-preflight checked

The launcher
`scripts/replay_go2_measured_plane_chained_single_pass_full_history_v1.py`
now connects the prepared exact-native input checker, complete paired replay,
saved-output reconstruction and chained timing accounting. Its prospective
protocol is
`docs/go2_measured_plane_chained_single_pass_full_history_v1_2026-09-12.md`.

The launcher requires the actual completed chained native result SHA-256,
ended original native owners, native/CPU serialization and the existing
64 GiB RAM / 43 GiB artifact-space dispatch thresholds. It creates one exclusive
output directory, records source/protocol/owner/input/resource identities,
executes and checks the complete pair, reauthenticates native inputs including
their original physical prefix, and binds all five output artifacts. It
preserves a terminal failure and rejects another invocation on the same output.
It does not implement a waiting queue or start a new native experiment.

Fourteen launcher tests passed in 2.78 seconds. They cover source-only preflight
without runtime admission/output creation, complete orchestration and real
hash binding of synthetic output files, missing-result/resource/busy/input
dispatch rejection, replay/checker/input-reauthentication failures, preservation
against retry, and CPU process identification without treating the current
launcher as a competitor. The earlier adapter, input and accounting tests
remain separately recorded; this test run does not constitute real sensor
replay or timing evidence.

The actual CLI `--source-preflight-only` completed with
`CHAINED_FULL_HISTORY_SOURCE_PREFLIGHT 2639`. At that check the original native
worker was still active, available memory was 66,883,145,728 bytes (below the
64 GiB runtime dispatch threshold), and artifact free space was
568,416,591,872 bytes. Source-only preflight observes hardware; it does not
admit an undersized runtime attempt. Resource admission will be checked again
after the original audit and owners finish.

| Prepared file | SHA-256 |
| --- | --- |
| `scripts/replay_go2_measured_plane_chained_single_pass_full_history_v1.py` | `17473a6fb686e8ed3421a23f3017e59b8170d2c7c59627f9d6e0e98f62f61c78` |
| `lewm/tests/test_measured_plane_chained_full_history_launcher_development.py` | `c85cc5e5ec92026424d85dfef326a0edf76feab75ce885465476ce4e9fde98ef` |
| `docs/go2_measured_plane_chained_single_pass_full_history_v1_2026-09-12.md` | `2d412e43182d1c75da36b5149e883d0279fbe590da47ba5a62ea65cc62457320` |

All 2,627 sources bound to native launch
`0f2963636abc4f1fc738e9323d2a74db02888ca7062dc1afa21023f8c443b8ff`
were independently rehashed and remain unchanged. The three new files are
outside that live binding. No timing attempt or queue was launched. The native
raw audit remains the current live dependency, and all navigation, timing and
hardware claims retain their previous scope.
