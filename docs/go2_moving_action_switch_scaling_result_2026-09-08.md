# Moving-action switch scaling result

The separately frozen native benchmark completed all eight executions. All four
serial/four-worker pairs match exactly across 310 saved array/pixel signatures
each (1,240 comparisons). Every execution completed 63 commands, 64 RGB-D
frames and 3,900 physics samples, with passing raw sensor/command/stop audits,
strict physical visibility and no hard measurement failures. No physical stop
occurred in these four fixed left-arc-to-hold cases.

Serial phase time was 202.5353366251 seconds; four-worker phase time was
58.5410957050 seconds, a 3.4597120909 speedup. The frozen rule therefore selects
four fresh CPU workers for scientific collection. Maximum measured per-worker
RSS was 2,084,634,624 bytes serially and 2,069,614,592 bytes in parallel.

Largest measured complete episode artifacts were 112,847,016 bytes. The fixed
144-case estimate with 25 percent allowance and 512 MiB of receipt allowance is
20,849,333,792 bytes, below the 24-GiB scientific allowance. Launch resources
were 81,834,758,144 bytes available RAM and 84,711,366,656 bytes artifact space,
16 physical/32 logical CPUs, 0.2 percent CPU busy and idle GPUs. The only
competing Python process used about 2.7 MB. Scientific launch rechecks resources
and retains the 40-GiB reserve.

Artifact base is the owned `navigation_development_artifacts_v1` directory.

| Root / receipt | SHA-256 |
|---|---|
| `go2_moving_action_switch_scaling_v1_attempt_001/launch.json` | `11afadb8793164a69cbe6ebb26a12e7bc695887fc8f734acbf17e12a3fd295a0` |
| `go2_moving_action_switch_scaling_v1_attempt_001/result.json` | `b593dfe4b8924d9bf8cdd9e6e382a3772575e332444d56c7dbb8aed20031ef79` |
| `go2_moving_action_switch_scaling_serial_v1_attempt_001/result.json` | `305cc6ef4fa2187f99b36ddfe2199f31de2d91af7a99e768e191d430ec18cf0e` |
| `go2_moving_action_switch_scaling_four_v1_attempt_001/result.json` | `08028f83d22d2c96fb734434b5046c6aa72c3397f5de8efebca0ec532f18564a` |

The launch and result bind 838 source paths and the inherited native/input
identities. Eight focused schedule, censoring, complete-population, prefix and
concurrency-selection tests passed before launch. Benchmark episodes remain
excluded from learning and transfer evaluation. This establishes usable native
collection concurrency, with no trained model, navigation arrival, independent
maze, real-time or hardware-deployment result.
