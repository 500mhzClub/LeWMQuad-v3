# Training-only translation correction native result

Both fixed corrected-model runs completed collection and exact raw replay, with
all measurement gates passing and **zero verified arrivals out of two**. This
reused development layout supplies zero independent-maze evaluations.

| Fixed seed 2026091001 model | Terminal tick | Minimum goal distance | Terminal goal distance | Commands including drain |
|---|---:|---:|---:|---:|
| Full JEPA | 97 | 0.993341333 m | 0.994433557 m | 107 |
| Full direct | 42 | 1.105141894 m | 1.105141894 m | 52 |

Both terminated with
`NO_PHASE_CANDIDATE_SATISFYING_SURFACE_AND_NOMINAL_CONSTRAINTS`, followed by the
unchanged ten-command drain. Neither had a physical or acquisition stop.
Maximum observed XY pose error was 0.0021068214643736064 m in both cases.
All 107 JEPA and 52 direct complete command iterations exceeded 100 ms:
medians 520.922924 and 518.9596815 ms; maxima 735.536791 and 570.753788 ms.
These timings include two-case concurrency with physics paused during compute.

The readout verifies exact physics, policy, fast-gyro, RGB and observer/memory
prefixes. Corrected JEPA first changed its requested command at tick 30;
31 observations and 28 forecast banks were compared. Direct first changed at
tick 19; 20 observations and 17 forecast banks were compared. Every compared
bank equals the uncorrected bank minus the frozen training coefficients using
float32 XY subtraction; yaw and contact entries remain exact. The new JEPA/direct
pair shares 22 causal observations before its command difference at tick 21.
No affected future frame is included in those comparisons.

The eight-step planner changed 44 JEPA and 13 direct selections relative to its
first-step choice; 53 and 78 candidate forecasts passed the first nominal
segment but failed a later segment. JEPA minimum goal distance improved from
1.117830109 m in its uncorrected predecessor, but that is not an arrival or a
general JEPA advantage. Direct remains stopped near the initial obstacle.

Recorded terminal evidence identifies distinct vetoes. At JEPA tick 97, all
six first nominal segments pass, but each candidate has a front-right-foot
sample-bounds intersection. Those intersecting samples are classified as
measured floor; the non-floor/unknown partition has zero intersecting samples.
Neither the complete-grid-foot rule nor the single-retained-frame foot-patch
rule establishes complete nominal foot coverage for any candidate. This does
not establish that the unobserved part of the foot area is safe. At direct
tick 42, all articulated checks pass, but every first nominal segment fails
against the already observed cell [11, -2], with minimum clearances
0.446464049–0.449240626 m below the unchanged 0.45-m threshold.

The next bounded diagnosis should reconstruct the JEPA prefix and distinguish
missing coverage from coverage fragmented across retained observations. A
read-only subdivision witness can test whether complete smaller squares cover
the original foot square using the existing measured-pixel rules. No footprint
shrinkage, floor interpolation, unseen-area waiver or native retry follows
from this diagnostic alone.

Identities under the fixed navigation development artifact root:

- Probe `go2_training_bias_goal_probe_v1_attempt_001`:
  launch `023bffcc40f9390bf95966529ebb57b09889fc1bd8b67e7f412d936cd0c47937`;
  result `5e48672a074d2086d734d02844af499d49d0a57571b9148b433b65ad2bedcfb5`.
- Readout `go2_training_bias_goal_readout_v1_attempt_001`:
  launch `34f2bebb73886008511b462acd8995bb73831036ecc9ac259ed6f532166b26a5`;
  result `f5ad4ad07dd06db5783143d044216b5371d7703fcdd3a692962f40cde887636f`.
- Correction result
  `a425d3ab1398df9312663e35dee665e5800153bd60d29334ae71a1d95e4f21d5`;
  corrected transfer readout
  `79753350e317eb096cdfe676d2fc9804a4e1f4c2af8c936241ff75229f869bcf`.

Six focused reader/scope tests passed in 1.89 s. The exclusive readout reverified
all probe and predecessor artifact/source bindings before and after analysis.
Original model weights, correction coefficients and completed failures remain
unchanged. Independent layouts, exploration/backtracking missions, matched
reactive/nonpredictive and memory baselines, realistic timing and bounded
hardware evidence remain outstanding; the overall goal is not achieved.
