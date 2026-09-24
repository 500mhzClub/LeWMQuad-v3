# Full-history single-pass timing replay completed

The complete 3,124-observation learned native history was replayed by fresh
independent original and single-pass controllers. Every complete original
decision reproduced the recorded native decision, every normalized candidate
decision matched the original, and public inputs remained unchanged. All 3,098
raw model forecasts matched, with exactly 3,098 actual model forward calls per
controller. Both assigned model states remained unchanged without gradients.

Ten declared complete state checkpoints matched at frames 0, 3, 61, 122, 255,
511, 1023, 2047, 3071 and 3123. The comparison includes the actual outbound
arrival, turnaround tracking failure and terminal stop. State equivalence uses
the eleven explicit implementation-type normalizations recorded in the report;
motion and mission types are not normalized. This is fixed-history equivalence
evidence, not a new native trajectory or navigation recovery.

| Controller-observation timing | Original | Single-pass |
| --- | --- | --- |
| All 3,124 observations, total | 4,088.734 s | 1,916.724 s |
| Median | 1,198.971 ms | 571.770 ms |
| p95 | 2,514.665 ms | 947.537 ms |
| Observations exceeding 100 ms | 3,114 | 3,114 |
| All 3,098 forecast observations, median | 1,208.624 ms | 573.908 ms |
| All 3,098 forecast observations, p95 | 2,517.488 ms | 948.256 ms |

Total controller-observation time decreased by **53.1218%** on the identical
recorded history. Every forecast observation still exceeded 100 ms. The ten
fast terminal-stop observations do not establish a real-time control loop.
The candidate's median alone remains about 5.7 times the target, without sensor
acquisition time. Controller order alternates within the paired replay; the
report explicitly does not claim an isolated benchmark. Other native work
overlapped part of execution. Compare these paired times with each other,
not with the separately collected native pilot's timing distribution.

## Completed evidence

Root:
`/home/andrewknowles/RecoveryStorage/LeWMQuad-v3/.generated/navigation_development_artifacts_v1/go2_measured_plane_single_pass_full_history_v1_attempt_001`.

| File | SHA-256 |
| --- | --- |
| `result.json` | `094519edfa684a38ccf144644c7d3c5bc4f14c67b3c6cf660737d6c627ef7cc0` |
| `launch.json` | `a38fccef6002ca980ca65a47ca9c77079cabc2ab0431a1e07484531098d999a7` |
| `comparison.jsonl` | `34a9cf3b9df99b4c62f95cded087f1d4741a69872f336554f944fa70eee7dfc1` |
| `state_checks.json` | `879a8fa50a7ea6568e5a9fde6f20b43db628e6d46342795cd11304b7233eec39` |
| `resource_monitor.jsonl` | `0725aa0c5aceaaf868c69541dca4ead3dcd3fc134f69d14db2c696c2de9f95da` |
| `report.json` | `b65f6044eb6ff8ea15a17ca639a90df6fadfd1b7ce775ad901e00e89d91bce68` |

Status: `MEASURED_PLANE_SINGLE_PASS_FULL_HISTORY_V1_COMPLETE`.
The original owner PID 2933637, creation time 1789172064.29, ended. No
`failure.json` was present. The final result reports complete output/public
packet rechecking and original raw input reauthentication before and after
execution. Its measured wall time, including validation, is 9,941.452 seconds.

After completion, all 2,583 bound sources and all five result-bound artifacts
were independently rehashed and matched. The launch identity and source
closure equality were checked. The report was independently reconstructed
from all 3,124 saved comparison rows and ten state checks, and matched both
saved report copies exactly. This final independent check does not claim a
second controller execution or repeat the raw-public-packet reconstruction;
the original process performed that check, and the existing waiter performs
its own completed-child verification before releasing the queue.

The fixed native input result remains
`4ecd63cd96aff9277103609b3034ed87e736e1fff1bb3352113941ff2c26aa18`,
and corrected model identity remains
`56799c99e2bb1fd5e5a591009b931c5b2e04741d3a3744e2346ce9896a2160dd`.

## Next action and limits

At this result check, timing waiter PID 2924370 was live in completed-child
verification. Chained-controller waiter PID 2930187 remained live, and its
child root was absent. Do not bypass either original waiter or restart the
completed timing attempt. Let the original queue start the tracking replay
after its required completion checks.

The full-history evidence supports the single-pass optimization on this
recorded trajectory. It does not change the already queued tracking candidate
or the prepared chained native experiment, which explicitly excludes that
optimization. Combining them requires a separate, verified composition.
The immediate goal remains a fresh audited round trip with reliable tracking;
independent-maze replication, matched scientific comparisons, realistic timing
and bounded hardware evidence are still required. No navigation, real-time,
hardware, deployment or full-goal completion is claimed here.
