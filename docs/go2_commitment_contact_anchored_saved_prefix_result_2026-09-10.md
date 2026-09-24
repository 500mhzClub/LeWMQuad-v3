# Completed saved-prefix preparation for ordinary commitment contact

Session 72671 completed the fixed full-supervised expanded-model saved-prefix
check with result
`f20100955ca2e4bb39c91a9375cb270ff48731d5e721aacf8d5b314821381808`.
It consumed exactly observations 0–3. The first changed command is at frame 3:
original left turn `[0,0,0.45]`, expected candidate forward `[0.2,0,0]`.
No observation 4 was consumed by this check. No model inference, candidate
controller execution or native trajectory was performed.

The original model remains the fixed expanded-data supervised assignment
`755c074325af96d53649aba4927937113d8fab561ea05341a2f3c328598b2bb5`.
The original collection is complete but its full raw audit remains pending.
This preparation does not substitute for that audit or a fresh paired replay.

The candidate reuses the previously checked commitment-contact scorer inside
the original anchored controller. It applies only to ordinary intermediate
waypoint choices after the original selection/recovery chain. Original active
residual recoveries, views, final goals and nominal-clearance reentry retain
their prior behavior. Only the ordinary soft contact-cost horizon changes from
800 ms to the actual 100-ms commitment; coefficient 1.2 and all eight original
800-ms geometry vetoes remain. Late contact scores remain evidence, but their
full-horizon cost is not charged in these ordinary choices. This is a policy
tradeoff, not an implementation-only optimization or physical-safety proof.

At the first boundary, all six original actions are phase/geometry eligible.
The recorded pose benefit and contact terms explain the changed ranking:

| Action | Pose benefit m | Contact score 100 ms | Contact score 800 ms | Original utility m | Candidate utility m |
| --- | ---: | ---: | ---: | ---: | ---: |
| Hold | 0.0107934 | 0.0001305 | 0.0032000 | 0.0069534 | 0.0106368 |
| Forward | 0.0250062 | 0.0004091 | 0.0461279 | -0.0303473 | 0.0245152 |
| Left arc | 0.0214727 | 0.0003100 | 0.0223665 | -0.0053671 | 0.0211007 |
| Right arc | 0.0219818 | 0.0003361 | 0.0314021 | -0.0157008 | 0.0215785 |
| Left turn | 0.0099797 | 0.0001243 | 0.0023572 | 0.0071510 | 0.0098306 |
| Right turn | 0.0108875 | 0.0001347 | 0.0039742 | 0.0061185 | 0.0107259 |

Neither contact score is a calibrated probability. The earlier old-model
commitment-contact maze-01 pilot produced translation but failed on current
visual evidence at observation 140, without an edge crossing or goal arrival.
That result remains a limitation, and a successful new trajectory is unproven.

Validation completed:

- Session 93128: 42 original-scorer and new-integration/comparator tests passed
  in 2.24 seconds, including preserved recovery/stop behavior and complete
  decision mismatch rejection.
- Session 64997: eight saved-checker tests passed in 2.50 seconds, including
  a generator that rejects consumption after the first changed command.
- Session 89417: source-only preflight passed 1,916 source bindings.
- Session 72671: the exclusive saved check completed and retained all four
  row/selection identities and the complete expected boundary selection.
- Session 42734: independent source/output/input binding verification and all
  four original saved row identities reconstructed. All six utilities were
  checked from saved pose-progress components and raw contact logits to an
  absolute tolerance of 1e-12 m; feasible argmax and every non-score selection
  field were checked. This did not rerun the model or candidate controller.

Verification JSON:
`docs/go2_commitment_contact_anchored_saved_prefix_verification_2026-09-10.json`,
SHA-256 `4615f55d77d447be427536b59672e7af500a89884b86f2721d5b9ceacc28794a`.

Exclusive artifact root:
`go2_commitment_contact_anchored_saved_prefix_v1_attempt_001` under the existing
recovery-storage navigation artifact root.

| Artifact | SHA-256 |
| --- | --- |
| launch.json | 278c303ad880197bda9abe751432e4d53a248b979f828f005798d84c784be6cb |
| comparison.jsonl | 3019b842ec0b9902d04c17c3cad5a4381cc7aee3fbde046b52b858c285234298 |
| first_boundary.json | d4068e3673165e170bf17f07e47cac7d772baa3ef83793927944b050b9a9a0f0 |

Next implement a separately named four-observation paired raw replay against
the completed original supervised worker, requiring its exact terminal/audit
identity and full input admission. Use fresh independent models/controllers;
verify both public input ownership and full original decisions, the exact
saved candidate selection, retained observed contact/map state and causal
residual/history evidence. Stop before observation 4. Only a completed raw
replay can support preparing a fresh native trial with physical-prefix and
complete raw outcome checks. No such native trial or replay waiter is launched
by this preparation, and existing native queue ordering remains unchanged.
