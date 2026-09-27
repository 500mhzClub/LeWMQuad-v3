# V3 corrected-binding screen result

**8/10 round trips; all ten beacons reached. Zero pose loss, contacts, hard or operating-margin violations.** Four outcome versions used. The9/10 threshold remains unmet; second-episode and oracle gates are unstarted.

| Episode | Round trip | Simulated seconds |
|---|---|---:|
| 00/0 | True | 130.42 |
| 01/0 | False | 480.00 |
| 02/0 | True | 159.62 |
| 03/0 | True | 463.12 |
| 04/0 | True | 132.72 |
| 05/0 | True | 165.72 |
| 06/0 | True | 160.92 |
| 07/0 | True | 187.72 |
| 08/0 | True | 301.62 |
| 09/0 | False | 480.00 |

01: all705 stale-latch holds are removed, but return planning remains turn/recovery oscillation (13 selected holds in839 return plans;194 blocked-latch releases). It still times out.

09: the few latch releases change the trajectory, allowing beacon arrival at266.8s versus458.1s inV2. It subsequently stalls on return, with347 no-eligible-movement holds across the mission, mainly weak-view recovery. This is no longer merely a late-arrival timeout.

Full cohort wall time: 2.151h. The post-cohort projection is104.34/160h before the next tuning screen; storage fits. It must be refreshed before another version.

Next diagnostic: two exact command-prefix replays, up to230s in01 and330s in09, to inspect the unchanged tracker’s recovery witnesses;563simulated seconds including settling,30-minute wall cap,16GiB RAM,32MiB retained scalars, no retained sensor frames, no controller rerun. Preserve all source inputs and any replay mismatch.
