# Camera V5 environment recovery V1 preregistration — 2026-07-16

## Boundary

This document preregisters one operational recovery at `.generated/go2_shared_observable_camera_ray_jepa_v5/protected_camera_adaptation_v5_native_schedule_completion_environment_recovery_v1`. It does not authorize deletion, reuse, resume, or mutation of the terminal V5 root. Exact recovery source closure, independent source review, a separate one-attempt execution authorization, an absent recovery root, and the visibility preflight below are required before reservation.

## Why a recovery is legitimate

The terminal environment-failure audit is `docs/lewm_go2_shared_jepa_v5_protected_camera_adaptation_v5_environment_failure_terminal_audit_2026-07-16.json`, file/content SHA-256 `3bfd02b66221dd54a4683e6d1836d3a55bf7ceff8f7a02b9e9f3d580b864d7c9` / `f7b3ce34f594547acc054b0e777fc24753d4e4092e7fa725e9eb363d76dbcfa7`. It binds an exact two-file terminal root: reservation file SHA-256 `0c3e538c79025dadfd65a5b31b8738293c673f3d2c8e499feb87e4f24a814989` and failure file SHA-256 `489a2744b2acdd3985e6a8e3d877ffb2b18c9abb15fcb2c494cda8867e56b0f2`.

The attempt stopped at `update0_reconstruction` with `PermissionError: matched training requires exactly one visible GPU`. It performed zero optimizer constructions or steps, training updates, Camera or JEPA objectives/backwards, EMA updates, physical evaluations, snapshots, metric-sidecar publications, GPU training operations, development-RGB opens, G2/navigation operations, and held-out opens. It produced no scientific or numeric evidence and qualified no checkpoint. The old root remains preserved and terminal.

## Exact science and control reuse

The recovery must exact-load the committed V5 contract, runner, and tests from source commit `6d001171d3f79fd8703e449272416191aae0c8b5`, with respective file SHA-256 values `ee732e692823b3bd9e3ac1c36611c976f8961cf6f6cc694cd82d05652351b582`, `3640ca35300ca36485487d6529dd352c76900c47018f7043cb165a1a078d72c4`, and `b835207f046c099f6a2450c51fe55c4a8bcf730d3f486ed1c9866e55e39cb767`.

The V5 science-contract SHA-256 remains exactly `d5f1ae7da90c505aca4fb6f0bc10c382d7d2a223ba6217b0b89b608a6dd1da76`; the control-contract SHA-256 remains exactly `3c7b72318aef6cdec2be4fa4e4c627e1a607b7685d3466dcac4a8ed2f41bd6be`; and the reporting-contract SHA-256 remains exactly `cb9eb1d162b97d2005d552d4189234965a8b4b5b7e1bf6a3a82559601f2d2eed`. The training-science change count relative to V5 is zero.

There is no architecture, loss, loss coefficient, data, refinement, sampling, schedule, seed, initialization, optimizer, learning-rate, clipping, threshold, checkpoint-control, evaluator, scope, margin, or publication change. The same fixed checkpoints remain updates 100, 400, 1000, 4000, 6000, and 8000. Observers still read only completed mode-0444 metric sidecars; `.pt` existence is not a readiness signal, checkpoint loading is forbidden, and evaluation reruns are forbidden.

The only operational changes are the fresh output root and the required pre-reservation visibility evidence. Recovery source may be a thin exact-hash wrapper around V5; it must not copy or alter the training loop.

## Required visibility preflight

Immediately before the recovery launch, with all six native thread variables equal to `1`, a separate no-tensor/no-checkpoint Python `-I -B` probe under `HIP_VISIBLE_DEVICES=0` must establish all of:

- `torch.cuda.is_available() is True`;
- `torch.cuda.device_count() == 1`;
- visible device zero's normalized name contains `r9700`;
- `ROCR_VISIBLE_DEVICES`, `CUDA_VISIBLE_DEVICES`, and `HSA_OVERRIDE_GFX_VERSION` are absent;
- no KFD training process and no other `.generated` mutator is active;
- the recovery output root is absent.

The probe opens no model, checkpoint, dataset, RGB, selection, navigation, or held-out artifact. If it fails, the recovery runner must not be launched and no output root may be reserved. No GPU-management query or competing GPU process may be started between a passing probe and launch. The execution authorization must bind the exact passing probe values and launch environment.

## Denials

Exactly one recovery attempt may be authorized. No retry, resume, second seed, schedule extension, model/data/loss change, threshold relaxation, soft promotion, automatic successor, JEPA training, G2, navigation, runtime use, held-out access, or held-out tuning is authorized by this preregistration. A failed recovery root must be preserved and terminal.
