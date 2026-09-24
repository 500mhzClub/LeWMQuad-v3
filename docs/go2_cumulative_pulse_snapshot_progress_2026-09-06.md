# Receipt-bound cumulative-model snapshot preparation

`scripts/cumulative_pulse_snapshot_development.py` adds exclusive checkpoint
persistence and verified evaluation-only reloads for the prospective matched
independent-layout study. No recorded-data fit, checkpoint selection, navigation
action or collection change is performed by this preparation.

## Implemented contract

Each snapshot binds the caller's experiment, dataset and schedule SHA-256 values
and named input variant. Reload also requires the exact expected objective,
model seed, latent width, learning rate, EMA momentum and completed update count.
A matching tensor hash alone cannot substitute for these identities. The variant
name is an identity binding, not an implementation of the input ablation itself.
Full source/data provenance and the meaning of the experiment manifest remain
the prospective experiment runner's responsibility.

Writes require an explicit existing owned development output root and an ordinary
snapshot filename. Files are exclusive, never overwritten, capped at 64 MiB,
and subject to the 40 GiB free-space reserve. Validated payload bytes are written,
flushed and fsynced, then loaded again and compared with the complete live trainer
snapshot. A failed write or reload does not delete or replace its evidence.
The outer experiment must record the failure and must not retry implicitly.

Reads verify the expected file hash before restricted CPU `weights_only`
deserialization, check the exact bytes actually read, and reverify the file after
loading. The loader requires the cumulative-event schema, integrated-hazard
semantics, fixed 6 cm position scale and a nonfailed, nonpromoted checkpoint.
It verifies the seeded initial model identity; every model tensor's exact key,
shape, dtype, finiteness and hash; every optimizer group/hyperparameter; and all
active AdamW states, moments and step counts. Zero-update checkpoints must still
equal their seeded initial model. The recreated model/optimizer states must agree
exactly after loading, without aliases to the supplied payload or original trainer.

Reloaded trainers are evaluation-only and reject optimizer-step requests. They
remain compatible with the streamed prediction interface but do not grant resume
authority or select a winning checkpoint. Every inference caller must still use
the planned data role, input treatment and common evaluation population.

## Verification

Initial 75782 completes exit 0: 36 new snapshot tests pass in 2.78 s. After adding
explicit payload-isolation, post-hash byte-change and fsync-failure tests, focused
91694 completes exit 0: **87 passed in 8.08 s**, comprising 39 new snapshot tests,
26 streamed-runner tests and 22 cumulative-model tests. Tests persist and reload
synthetic direct/supervised-rollout/JEPA snapshots and a zero-update snapshot,
verify exact forward outputs, reject corrupted identities/tensors/optimizer
states and verify retained write-failure evidence. Synthetic optimizer steps are
not recorded scientific training results.

Full 237-file regression 5127 completes exit 0: **3,150 passed in 240.28 s**,
including the 39 new snapshot tests. Read-only 76694 completes exit 0: all **771 launched supervisor source
bindings remain unchanged**, and the new snapshot source is outside that closure.

Snapshot-helper SHA-256:
`54b77a91de9dc4b1cbff63f35ed39cb52a38de2dfed89cf37886efb66a34a635`.
Test SHA-256:
`66502f2a534ec81d5436a83d5b0ac3e116df20ba2533cc75491551557704dfbc`.

## Live acquisition and next steps

Supervisor 25963 remains live as PID 2063013 with its l01 child PID 2063119.
The latest bounded metadata check observes 26/120 raw-prechecked l01 cases,
all eligible, one retained strict boundary-depth failure, no hard measurement
failures and no positive contact episode yet. Parent/child are independently
observed live at 890/839 s elapsed. l00 remains completed and audited; l01 and
the remaining sequence are not complete. Do not start a competing stage or
modify the 771 supervisor-bound sources.

Subsequent native output on the same live supervisor handle reaches at least
32/120 l01 cases raw-prechecked, through corner-turn/quiet/lower-friction action1.
This later live-progress observation does not extend the bounded aggregate
measurement counts from the 26-case metadata check above.

Next complete the explicit input-ablation treatments and freeze the actual
scientific experiment executable, receipts, coverage gates, matched seeds and
exposure budgets. Integrate this snapshot helper with per-update persistence,
failure accounting and scoring of reloaded primary/auxiliary heads. Keep the
empirical action/time and zero-motion baselines, exact row pairing, joint
contact/motion/future-image coverage and layout-level analysis. Do not claim
future-image latent supervision for censored collision cases.

The full goal still requires reliable local physical execution, demonstrated
predictive-training and online-rollout contributions, useful memory/backtracking,
whole unfamiliar-maze missions, real-time/deployment-valid sensing and bounded
hardware evidence when available. Snapshot verification is implementation
evidence, not success on those requirements.
