# Experiment attempt accounting policy V2

Date: 2026-08-27

## Purpose

This policy separates technical startup, scientific execution, and publication so a process-control failure is not silently counted as a scientific model attempt. It does not authorize an additional experiment.

## Attempt classes

- `technical_startup_attempt`: is recorded when the exact child command receipt, external stream and inherited-FD custody files, and the exclusive external diagnostic-root/path reservation are durably created before Popen; child creation then binds those prelaunch records. Complete diagnostic custody is finalized only after child exit. It is not a consumed scientific attempt.
- `scientific_attempt`: begins only after PREEXECUTION passes and the first scientific input or model is opened
- `publication_attempt`: begins when an independently validated immutable scientific payload enters the authorized durable publication/finalization path

## Mandatory rules

1. A failure before PREEXECUTION completion and before any scientific input or model open is not a scientific attempt.
2. Technical startup correction and publication-only correction are accounted separately from outcome-bearing scientific retries.
3. A publication-only correction may republish only an immutable, independently validated scientific payload; it may not recompute it.
4. Every child stdout, stderr, and traceback stream is preopened and retained outside the attempt namespace.
5. A final-retry limit is enforced only after diagnostic custody has qualified the failure phase and attempt class.
6. Any change informed by scientific outcomes requires a new experiment version and a new prospectively frozen scientific contract.
7. An identical technical recovery may not change models, inputs, targets, metrics, gates, seeds, or scientific decision logic.

This policy does not retroactively reclassify completed scientific experiments.

## Durable reservation boundary

- Technical: recorded by the pre-Popen exact command receipt plus preopened external stream/FD custody and exclusive diagnostic-root/path reservation; the complete diagnostic-custody receipt is post-exit and never consumes the scientific retry budget
- Scientific: consumed only after PREEXECUTION passes and the first scientific input or model is opened under its durable reservation/namespace
- Publication: recorded only when immutable-payload publication custody is durably reserved; it cannot recompute scientific content

A diagnostic invocation, syntactic rejection, child process start, or technical incident archive is not by itself evidence that a scientific attempt was consumed. Missing reservation evidence is handled fail-closed.

## Current incident

The third incident has exact technical receipts, no attempt or reservation namespace, and zero scientific counters. It is accounted as a technical-startup incident, not a scientific attempt. This incident accounting alone authorizes neither a V2 specification nor V2 execution. A V2 specification requires the complete frozen forensic gate and the bound current-user specification-only authority; V2 execution and a scientific attempt remain unauthorized.

## Authority boundary

The scientific contract remains frozen at `9c1c3adcfb8382c33e8da8895dc345e006e92e43` with digest `1667f325be2c835a6222dc90bb684f373a06b365d59b70e9746fd7adb052c382`. The present forensic overlay changes no model, input, target, metric, gate, seed, safety scope, or scientific decision.
