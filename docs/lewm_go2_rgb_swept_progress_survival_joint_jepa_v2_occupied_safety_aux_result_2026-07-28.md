# RGB Swept-Progress Survival Joint-JEPA V2 Occupied-Safety Auxiliary — Terminal Result

- Terminal status: `FAIL_FULL_ARM` (valid completed scientific run).
- Exactly one attempt completed on 2026-07-28; no retry, resume, or predecessor checkpoint use occurred.
- Result file/content SHA-256: `50f1752f96c711d2997fdc3d03a45123d8e798b78cd72edea72f0ea6c9993dec` / `75c66df767801a0359f4a6b61f494afe86bce52010bf6ba2410826dfdb9df191`.
- Training-trace file/content SHA-256: `507e725d8fa3a7a0e755f67d373e44f02bff12f65dc3a9b347146b3623f37023` / `a3fb7d606fbe8ed80c046fd1a83320b7f5e70547358853726a1b9704ad35a9b8`.
- Independent artifact audit: PASS for receipt integrity and correct scientific failure classification.

## Integrity and access

- Exact accounting: 1,000 updates, 16,000 presentations, 4,000 microbatch graphs/backward calls/predictor objectives, and 1,000 optimizer and EMA steps.
- All 1,000 trace rows are present, finite, contiguous, and include `O`; `L=S+P+U+R+O` verifies.
- Exact frozen input, schedule, N320, mask, seed, hardware, control, and gate receipts verify.
- Forbidden input count and G2/navigation/final-evaluation open count are all zero.
- The V2 checkpoint is rejected, unqualified, non-resumable, and has not been opened after the terminal result. The matched no-JEPA arm did not run.

## Scientific result

- The occupied-safety auxiliary fixed both V1 failures:
  - occupied recall: `0.644302 -> 0.777180` (V2 floor `0.70` passed);
  - rough occupied recall: `0.580587 -> 0.768724` (floor `0.65` passed).
- Balanced accuracy improved `0.822839 -> 0.855027`; unknown recall improved `0.938535 -> 0.949281`.
- V2 failed only free recall: `0.838621 < 0.85`, a `0.011379` absolute miss.
- The free-class confusion row is `[3695 unknown, 419817 free, 77092 occupied]`; `95.426%` of free errors are false occupied predictions. This is directly on the boundary emphasized by `O`, consistent with an over-conservative coefficient rather than a new representation failure.
- Every swept-progress and family gate still passed: utility `0.891266`, zero-prefix rate `0.035088`, and pair concordance `0.863039`.
- Every control still passed with positive bootstrap lower bound: persistence delta `+0.168706`, shuffled-action `+0.317880`, wrong-RGB `+0.090019`, and action-prior `+0.061257`.
- Training was stable but plateaued; a schedule extension is unsupported. First-100 to last-100 mean `O` changed `3.536120 -> 3.396272`, while the last window regressed relative to updates 801–900.

## Next decision

- Do not extend training, alter thresholds, select an intermediate checkpoint, warm-start, or sweep coefficients.
- One final bounded V3 is warranted because V1 and V2 bracket the required free/occupied trade-off and all navigation/control gates survived both endpoints.
- V3 changes only `O` coefficient `1.0 -> 0.5`, the natural midpoint between V1 (`0`) and V2 (`1`). Linear interpolation is diagnostic rather than a claim, but at `0.5` it projects free `0.862151`, occupied `0.710741`, and rough occupied `0.674655`, all above their fixed floors.
- An independent reviewer preferred a nonlinear semantic head over coefficient tuning. That larger mechanism is deferred because the observed V2 error moved directly along the registered occupied-vs-rest boundary and the user explicitly permits one obvious improving correction; V3 remains one midpoint falsification, not a sweep.
- Preserve all other V2 science, execution, controls, gates, and the 1,000-update / 16,000-presentation cap. Use a fresh model from accepted N320 only and never read either rejected predecessor checkpoint.
- This is the last coefficient attempt. If V3 does not pass every gate, close the safety-weight family. If it passes, run the matched no-JEPA arm before any JEPA treatment-effect, G2, navigation, or held-out claim.
