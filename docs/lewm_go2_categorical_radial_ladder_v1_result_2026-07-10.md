# Go2 categorical radial ladder v1 result

Date: 2026-07-10

Status: N=1 passed; N=4 terminal gate failed; N=16 not attempted

This is a train-role-only implementation diagnostic. It did not evaluate G2
and cannot promote a perception checkpoint.

## Immutable artifacts

- ladder manifest:
  `.generated/go2_categorical_radial_micro_overfit/v1/ladder_manifest.json`
- ladder manifest file SHA-256:
  `967812399045b29e8be316f2f87bc16f02d681b0ea01884513c6b4f29bbe4b12`
- ladder result:
  `.generated/go2_categorical_radial_micro_overfit/v1/seed_20260710_ladder_result.json`
- result file SHA-256:
  `72e4ecbe6b9e9024bb910e5231deb42e2d73f3187babd2a9af518251cbb7c2a2`
- result content SHA-256:
  `02c627eb01e42a5b7e8ea57e5bd4bde3d1fc2ca0667abdd9dd1cf8162beacd52`

The result content hash recomputes exactly. It binds the final runner, model,
factorization, protocol, preparer, encoder, and panel-contract source hashes.
Both the ladder manifest and parent panel remained unchanged through execution.
All checkpoint-selection, probability-calibration, G2, and other non-train
access counters are zero.

## N=1 result

N=1 consumed all 1,000 registered updates and passed:

- balanced hierarchical NLL: `0.00003419`;
- UNKNOWN/FREE/OCCUPIED recall: `1.0 / 1.0 / 1.0`;
- both hierarchical balanced accuracies: `1.0`.

The gate first passed at step 400 and remained passing at every subsequent
100-step evaluation through step 1,000. This proves that the categorical
radial factorization, image projection, loss, gradients, and decoder can
exactly reconstruct a complete all-class frame.

## N=4 result

N=4 consumed all 1,500 registered updates. It passed the complete gate at
every evaluation from step 300 through step 1,400, then failed at the mandatory
terminal step 1,500.

At step 1,400:

- balanced hierarchical NLL: `0.00088808`;
- UNKNOWN/FREE/OCCUPIED recall: `0.99948 / 1.0 / 1.0`;
- the wrong-view NLL separation was over 5.0.

At step 1,500:

- balanced hierarchical NLL regressed to `0.00743824`, still below the 0.01
  threshold;
- UNKNOWN/FREE/OCCUPIED recall became
  `0.99525 / 0.99888 / 0.98387`;
- occupied recall therefore missed the fixed 0.99 threshold;
- wrong-view NLL remained strongly separated at `5.40202`.

The preregistered rule evaluates the fixed terminal checkpoint, so this result
is a failure. N=16 was correctly not run. The long passing interval and final
excursion identify optimizer-tail instability rather than a coordinate,
grounding, or representational-capacity failure at four frames.

## Next admissible change

The next ladder version must keep the frozen manifest, model architecture,
initialization rule, loss, update counts, controls, and gates unchanged. It may
change only the precommitted learning-rate schedule to suppress the terminal
AdamW excursion. No result from steps 300-1,400 is promoted or used as an early
stop, and no non-train role may be opened.
