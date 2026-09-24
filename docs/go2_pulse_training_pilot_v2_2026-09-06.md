# Pulse-training pilot V2: science-identical persistence correction

Preserve original V1 output and frozen sources. V1 stopped on an exclusive
JSON writer conflict: update1 was saved, then update2 executed but its attempted
rewrite of updates.json raised FileExistsError. No fit/final checkpoint was
completed. This is an infrastructure failure, not a failed model comparison.
Original launch SHA0123e2c16588abfcf8c6c3cdf5f77161c75da3b2f2bc77d104887ff3933904bd;
failure SHA68f4848b18a8569f9e811f71251addf131f78ae716fc59ac812fa8d817119070.

The distinct V2 launcher changes persistence only: write one immutable
update_NNNN.json per optimizer step and the aggregate updates.json once after
all12 updates. Test this with the actual exclusive writer and checkpoint reload,
including refusal to reuse an output directory. Do not modify V1, overwrite
its log, resume its optimizer or choose hyperparameters from its one saved loss.

All science and resources from the [V1 protocol](go2_pulse_training_pilot_v1_2026-09-06.md)
are unchanged: seeds2026090721–2026090723; direct, supervised-rollout, JEPA;
latent32; matched fixed12x6 schedules; AdamW.001/no decay; clip1; post-step
EMA.99; CPU deterministic one-thread execution; nine fresh fits/108 updates;
all185-window before/after train-role resubstitution and zero-motion control;
fixed final checkpoint only;1GiB allowance plus10GiB reserve. No GPU, physics,
best-checkpoint selection, additional seed, data change or scientific retry.
This is a new integrity-corrected attempt, not a resume of the original.

Exclusive `.generated/go2_pulse_training_pilot_v2_attempt_001`. Bind original
launch/failure/one saved update and source closure, complete dataset/input/raw
bindings, corrected launcher, actual persistence test and this protocol.
Freeze before launch; verify before/after. Preserve failures and every update.
No source export, sealed material, old-output changes, model promotion or
navigation use. Goal completion still requires independent maze/layout studies,
useful learned control and online memory, realistic deployment-valid sensing,
reliable physical execution and bounded hardware evidence when available.
