# Paired-floor V1 development result

**5/10 round trips and 9/10 beacon retrievals**, versus 3/10 and 5/10 in C3. The 9/10 screen threshold is not met; the oracle gate remains unstarted.

| Episode | Beacon | Round trip | Outcome | Simulated seconds |
|---|---|---|---|---:|
| 00/0 | True | True | round trip | 130.42 |
| 01/0 | True | False | timeout | 480.00 |
| 02/0 | True | True | round trip | 159.62 |
| 03/0 | True | True | round trip | 463.12 |
| 04/0 | True | True | round trip | 132.72 |
| 05/0 | True | False | timeout | 480.00 |
| 06/0 | True | True | round trip | 160.92 |
| 07/0 | True | False | pose loss | 249.00 |
| 08/0 | False | False | timeout | 480.00 |
| 09/0 | True | False | timeout | 480.00 |

All ten missions: zero disallowed contacts, hard-criterion violations, operating-margin violations, or unresolved sampled hard clearance. 02 and 06 become successful round trips; 01 and 09 now retrieve the beacon but time out on return. 08 remains an outbound timeout. 05 remains a return timeout and 07 retains its pose-loss failure.

The output contract passes 20/20 against evaluator-side native ground truth with the tolerance frozen at 2 mm before new-path comparisons. Maximum absolute V1 error is 0.305 mm. C3 passes only 10/20 under the same contract.

All five normal-start recordings (00,03,04,05,07) reproduce complete native arrays, requests and consumed sensor hashes exactly. The fix nevertheless changes normal-start scalar floor initialisation, so the version charge stands: two of six consumed.

Screen wall time: 2 h 34 min. Startup contract and recording comparisons: 8 min 10 s. Latest projection after this cohort: 79.97/160 wall-hours with 15% contingency; another tuning screen needs a new projection.

No next change or further simulation has been launched. Next: diagnose the four remaining timeouts alongside the unchanged pose loss. Tracker estimation remains unchanged; any estimator change requires approval.

## Full startup-output contract

# Paired-floor startup-output contract

Status: **PASS**. Tolerance fixed before new-path tests: 2.0 mm.

C3 passed 10/20; paired-floor V1 passed 20/20.

| Episode | C3 error (mm) | V1 error (mm) | V1 pass |
|---|---:|---:|---|
| 00/0 | -0.774 | -0.043 | True |
| 00/1 | 244.260 | -0.170 | True |
| 01/0 | 234.042 | 0.108 | True |
| 01/1 | -0.506 | 0.018 | True |
| 02/0 | 240.169 | -0.229 | True |
| 02/1 | -0.491 | -0.035 | True |
| 03/0 | -0.758 | -0.008 | True |
| 03/1 | 113.702 | 0.073 | True |
| 04/0 | -0.996 | -0.043 | True |
| 04/1 | 233.176 | 0.041 | True |
| 05/0 | -0.782 | -0.006 | True |
| 05/1 | -0.607 | 0.013 | True |
| 06/0 | 290.184 | -0.166 | True |
| 06/1 | 266.448 | -0.305 | True |
| 07/0 | -0.482 | -0.058 | True |
| 07/1 | -0.165 | -0.051 | True |
| 08/0 | 260.113 | -0.167 | True |
| 08/1 | 238.127 | -0.059 | True |
| 09/0 | 286.535 | 0.090 | True |
| 09/1 | -0.766 | 0.014 | True |

Normal-start recording comparison:

- 00/0: exact=True; first sensor difference frame None; first applied-command difference index None
- 03/0: exact=True; first sensor difference frame None; first applied-command difference index None
- 04/0: exact=True; first sensor difference frame None; first applied-command difference index None
- 05/0: exact=True; first sensor difference frame None; first applied-command difference index None
- 07/0: exact=True; first sensor difference frame None; first applied-command difference index None

The version charge stands: V1 changes normal-start floor initialisation too. Two of six versions remain charged. No running-screen code, assignments or criteria were changed.

Any failed or unresolved start requires reporting before a further change. No tracker estimation change was made.

Full evidence: `/home/andrewknowles/RecoveryStorage/LeWMQuad-v3/.generated/navigation_development_artifacts_v1/go2_navigation_capability_v1_attempt_001/paired_floor_output_contract_attempt001/result.json`.

