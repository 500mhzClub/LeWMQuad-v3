# Camera-ray N5 gate-aligned raster NLL V15 implementation handoff

Date: 2026-07-14

Implementation author: `/root/camera_v12_gate_aligned_implementer`

Status: **source and synthetic CPU closure complete; terminal-V14 proof
clarification bound; independent review and a reviewed source-closure commit
required; no GPU diagnostic or exact authority**

## Frozen source authority

V15 is governed by the source-free runtime-visibility amendment:

`docs/lewm_go2_observable_camera_ray_fit_v4_n5_gate_aligned_raster_nll_v15_runtime_visibility_successor_amendment_2026-07-14.md`

File SHA-256:

`b1809b74cd400f8c56b5a912017c9466bb69aa0a7f4e390ccd3be59492a0f393`

It also binds the source-free terminal-V14 proof clarification:

`docs/lewm_go2_observable_camera_ray_fit_v4_n5_gate_aligned_raster_nll_v15_terminal_v14_proof_clarification_2026-07-14.md`

File SHA-256:

`2e4bae3e4cd33d6b62e006b8961aa9e212a8adb40272f624a26cffe876bb27f4`

The clarification changes only the retained V14 proof expectation. It grants
no experiment, GPU, training, navigation, or held-out authority.

The amendment author is `/root/camera_v15_runtime_contract`. A canonical V15
reviewer must start with `/root/` and differ from `/root`, the amendment
author, this implementation author, the V12 and V13 reviewers, the V14
reviewer `/root/camera_v14_independent_review`, and the future exact execution
agent. The implementation author did not write a canonical V15 review.

## Frozen V15 closure

V15 adds no model or loss file. It retains
`lewm/models/observable_camera_ray_evidence_v4_gate_aligned_raster_nll_v12.py`
at SHA-256
`735563f811c5d7b9efb9e37dca8348825a8467bd0a059f83ab94d41d45d57662`.

### Production files

| Role | Path | File SHA-256 |
| --- | --- | --- |
| V15 policy | `lewm/benchmarks/go2_observable_camera_ray_fit_v4_n5_gate_aligned_raster_nll_v15.py` | `17677435731779c9549b5fb8f08b3268f223bc7a945d40f4f2f572a3b652e0ed` |
| standalone GPU-visibility preflight | `scripts/preflight_go2_observable_camera_ray_fit_v4_n5_gate_aligned_raster_nll_v15_gpu_visibility.py` | `fe913bd04448ea5ddae39186c805c8448c72a4f0bd12b430c26dd29a991b3051` |
| V15 trainer | `scripts/train_go2_observable_camera_ray_fit_v4_n5_gate_aligned_raster_nll_v15.py` | `62f5d9d5072bb83f6c8fd9af4c8bb32a96357d3365ba87a5258a529ae1ddcaf1` |
| V15 verifier | `scripts/verify_go2_observable_camera_ray_fit_v4_n5_gate_aligned_raster_nll_v15.py` | `bb3e8838689105ab2ee1e4e5525d1de341525439aa83e526ff834efce89a1584` |
| V15 executor | `scripts/execute_go2_observable_camera_ray_fit_v4_n5_gate_aligned_raster_nll_v15.py` | `8879a42bd091609e4d48aa8ff743d0ab5adcb595caead3507c4393afcc8a7d6d` |

### Proof files

| Role | Path | File SHA-256 |
| --- | --- | --- |
| synthetic lifecycle and native gate fixture | `lewm/tests/n5_gate_aligned_raster_nll_v15_synthetic_execution.py` | `19d1f6e18d143bd5e62bac8e1d9a06a1ee21c2f31aae48187e0cc9cc4074333c` |
| loss, parity, gradient, and diagnostic tests | `lewm/tests/test_go2_observable_camera_ray_fit_v4_n5_gate_aligned_raster_nll_v15.py` | `0b003280bad8240e122c8cd6e51ad9d2a7be6a135629b7702876c265176fe18b` |
| lifecycle, authority, replacement, and subprocess tests | `lewm/tests/test_go2_observable_camera_ray_fit_v4_n5_gate_aligned_raster_nll_v15_lifecycle.py` | `c71c830e8e36e9b8904e7dd29190fa3864f7f20d7cb71cec40032b790c678cd2` |
| runtime-visibility and receipt tests | `lewm/tests/test_go2_observable_camera_ray_fit_v4_n5_gate_aligned_raster_nll_v15_runtime_visibility.py` | `c66f0e11563795a97be9393f04bd489fa63ae23856f8776fb80ad2c9a2ad6d9c` |

This handoff was written after the hashes above were frozen. It does not
self-hash; the independent reviewer must hash its final bytes and bind that
hash in the canonical review.

## Bound terminal V14 state

V15 binds the reviewed V14 source closure at Git commit
`3cb1e2f7316493dd62c9d44bd9878df6d1f6a0c6`. It binds the V14 review written by
`/root/camera_v14_independent_review` with file SHA-256
`19f0082b587d94e50b7c4fac38e11f1af35bcd76ae1acaaf0e44c5ad2721e2ac`
and canonical content SHA-256
`54c8c12953849e3515c4cb73e945033dcd3532450884500a94685cc02785a243`.

The terminal V14 lifecycle evidence is frozen as follows:

| Role | Path | Bytes | File SHA-256 | Content SHA-256 |
| --- | --- | ---: | --- | --- |
| reservation | `.generated/go2_observable_camera_ray_fit_v4/n5_gate_aligned_raster_nll_v14/attempts/seed_20260710/n5/reservation.json` | 23,771 | `56abce14d8ba7901103bbd23353095c30180ca5361f7595b178da6e440ecea8c` | `0aabc7ac8a468c6524ba66a244a11126c8e2c1d7587dbc1fafb3de71cc7d443b` |
| failure | `.generated/go2_observable_camera_ray_fit_v4/n5_gate_aligned_raster_nll_v14/attempts/seed_20260710/n5/failed.json` | 1,296 | `df6d91925fb167bc72e41eb9a6f07657f246c6a7a95d3bf20734c747e639c704` | `79560d4b5532ff41e428da913e28c6db235608da6cf4ea107fd33207870afad7` |
| zero-byte lock | `.generated/go2_observable_camera_ray_fit_v4/n5_gate_aligned_raster_nll_v14/attempts/seed_20260710/.n5.reservation-v14.lock` | 0 | `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` | n/a |

The V14 attempt remains consumed and terminal. The exact terminal evidence was
opened read-only through no-follow proof readers solely to establish the
clarified replacement proof. No V14 source, proof, review, reservation,
failure, lock, directory, or other generated byte was altered, deleted,
renamed, repaired, or reinterpreted.

## Implemented runtime-visibility delta

The scientific experiment remains V14-identical after namespace
normalization: seed `20260710`, fit size five, retained V12 model/rasterizer,
the four retained V11 losses plus exact additive
`0.25 * derived_raster_cell_nll`, schedule SHA-256
`fb5a6c13708944b6ce514960c11c063eece704676276b352bd5233080f4fd380`,
4,000 batch-five updates, 20,000 exposures, final-update selection, matched
and cyclic wrong-RGB controls, native diagnostics, isolated verification, and
all 26 checks and thresholds.

The additive V15 runtime delta is:

1. A standalone, import-light visibility process classifies exactly one of
   `pass_exactly_one_r9700`, `gpu_runtime_unavailable`,
   `gpu_device_count_mismatch`, `gpu_device_identity_mismatch`,
   `gpu_selector_mismatch`, `native_thread_mismatch`, or
   `gpu_visibility_receipt_publication_failure`. A successfully enumerated
   zero-device result is distinct from an unavailable runtime. Before its
   first repository import, even a natural non-`-B` CLI invocation sets
   `sys.dont_write_bytecode = True`; dispatch then relaunches the diagnostic with
   `-I -B`. Thus the bootstrap cannot create repository `__pycache__` or
   `.pyc` files before isolation.
2. The passing predicate requires only logical `cuda:0`, exact device name
   `AMD Radeon AI PRO R9700`, `HIP_VISIBLE_DEVICES=0`, all conflicting
   selectors and `HSA_OVERRIDE_GFX_VERSION` absent, all six frozen native
   thread variables equal to `1`, and PyTorch intra/inter-op counts equal to
   one.
3. The diagnostic performs enumeration only: no tensor allocation, kernel,
   data/RGB/checkpoint/output open, `.generated` open, or reservation.
4. Its canonical external receipt path is fixed at
   `/tmp/lewm_go2_observable_camera_ray_fit_v4_n5_gate_aligned_raster_nll_v15_gpu_visibility_preflight_2026-07-14.json`.
   Publication uses a private sibling and Linux
   `renameat2(RENAME_NOREPLACE)`. Validation requires canonical self-hashed
   JSON, a no-follow singly linked mode-`0600` regular file owned by the
   reader, exact review and Git-closure bindings, the same host and boot, and
   UTC and monotonic freshness no greater than 600 seconds. No alternate path
   is accepted. The reviewed Git closure accepts only a plain exact string
   containing lowercase hexadecimal of length 40 for a SHA-1 repository or 64
   for a SHA-256 repository. Every reviewed source/proof and both source
   authority documents are still rehashed from that exact Git object before
   the commit is accepted.
5. Exact execution validates static authority and the caller-bound V15 review,
   then the caller-bound fixed receipt, then repeats the live predicate, then
   proves `.generated` mutator quiescence and V15 output freshness. Only after
   all checks pass may it publish the reservation, which is the first V15
   output mutation. Data, provenance, RGB, model, and checkpoint access remain
   after the durable reservation.
6. The reservation binds the receipt file/content hashes, review, reviewed Git
   commit, and live observation. Exact execution refuses a non-isolated
   invocation instead of silently relaunching and must be invoked explicitly
   with Python `-I -B`.
7. Every post-reservation outcome consumes the sole V15 attempt and remains
   terminal with no retry. Every pre-reservation visibility rejection leaves
   the V15 output root absent and consumes no scientific attempt.

### Receipt-publication failure boundary

There is one unavoidable representation boundary in the amendment wording:
if all observations pass but publication of the canonical receipt itself
fails, the process cannot also durably write the
`gpu_visibility_receipt_publication_failure` disposition to that same fixed
path without contradicting the observed publication failure or overwriting a
pre-existing receipt. In this case V15 emits a canonical, self-hashed failure
receipt to stdout, returns nonzero, and leaves the fixed path untouched. That
stdout object is diagnostic evidence only, is never accepted by the exact
launcher, and grants no attempt or downstream authority. The CPU proof suite
exercises this exact boundary.

## Terminal-V14 proof replacement

The original retained V14 suite contained a pre-attempt assertion that the
V14 root was absent. That historical predicate became false when the one V14
attempt durably published its terminal lifecycle evidence. V15 does not edit
or work around the immutable test.

Under the bound clarification, the retained V14 run deselects exactly:

`lewm/tests/test_go2_observable_camera_ray_fit_v4_n5_gate_aligned_raster_nll_v14.py::test_v14_sources_forbid_v11_checkpoint_input_and_exact_root_is_absent`

The V15 lifecycle test
`test_v15_terminal_v14_state_replaces_obsolete_root_absence_proof` replaces
only that predicate. It no-follow walks the exact V14 root; requires the exact
three-file terminal inventory; rechecks byte counts, file and content hashes,
canonical JSON, seed, fit size, attempt count, source review, runtime failure,
cleanup journal, `retry_authorized=false`, and all absent checkpoint/result/
completion/metric/gate outputs; and requires the zero-byte lock at its frozen
hash. The resulting retained expectation is exactly:

`V14: 234 passed, 1 exact pre-attempt-absence node deselected, 1 terminal-state replacement proof passed`.

## Proof results

All harness commands hid accelerator selectors, removed
`HSA_OVERRIDE_GFX_VERSION`, set all six frozen native thread variables to one,
disabled bytecode and external pytest plugins, and used CPU-only synthetic
temporary roots. The focused bootstrap regression intentionally removed
`PYTHONDONTWRITEBYTECODE` only from its natural CLI subprocess in a temporary
repository copy. No real visibility diagnostic ran and the fixed `/tmp`
receipt was not touched.

The full V15 closure ran the science, lifecycle, runtime-visibility, and
frozen ladder-gate suites. Result: **284 passed in 19.86 seconds**, partitioned
as 23 science, 195 lifecycle, 46 runtime-visibility, and 20 ladder-gate tests.
The affected runtime-visibility suite passed **46 tests in 0.24 seconds**.

The new focused regression copied only the standalone preflight and policy
into a writable temporary repository, removed all ambient bytecode suppression,
and invoked the natural CLI without `-I` or `-B` using `--help`. Both bootstrap
and isolated-child imports completed with no `__pycache__` directory or
`.pyc`/`.pyo` file anywhere in that copy. It passed alone in **0.16 seconds**.

Three focused Git-object tests passed in **0.10 seconds**. They rehashed the
complete reviewed closure from the repository's actual 40-character SHA-1
HEAD, exercised a synthetic 64-character SHA-256 object ID through the same
containment function, and rejected nonstrings, booleans, string subclasses,
wrong lengths, uppercase, and nonhex input. The actual-HEAD proof used only
`git rev-parse` and `git show`; it performed no GPU, receipt, data, or output
operation.

The clarified retained V14 closure passed **234 tests with the one exact node
deselected in 18.87 seconds**. The terminal-state replacement proof passed
alone in **0.21 seconds**. Retained closures also passed:

- V13: **226 passed in 25.51 seconds**;
- V12: **202 passed in 25.60 seconds**; and
- V11: **190 passed in 24.33 seconds**.

All nine V15 Python production/proof files compiled in memory without
bytecode. They have LF line endings, a final newline, and no trailing
whitespace. The full suite passed normalized-AST, frozen-hash, source-review
open-order, fixed-path, zero-access, first-mutation, lifecycle fault-injection,
author-separation, no-authority, and output-absence checks.

The production executor's real isolated CPU verifier smoke also passed. It
validated all 11 verifier phases, timeout/signal/nonzero/malformed/oversized/
stderr process cases, independent V15 raster-NLL and native diagnostic
recomputation, the shared 26-check gate, exact/smoke schema separation, removal
of its temporary tree, and zero publication.

## Access and authority closure

During implementation and proof:

- no canonical experiment data, RGB payload, checkpoint, metric, or numeric
  result was opened;
- the only canonical experiment `.generated` payload reads were the three
  immutable V14 terminal entries required by the clarification; none was
  mutated;
- no V15 output path was created or mutated;
- no GPU or iGPU runtime operation, device enumeration, tensor allocation, or
  kernel ran;
- the real external diagnostic did not run and its fixed `/tmp` receipt was
  not created, removed, replaced, or read;
- no V15 training, exact verification, finalization, publication, benchmark,
  G2, navigation, or held-out evaluation ran; and
- no canonical V15 independent review was written by this implementation
  author.

V15 currently grants no exact authority. It also grants no V14 retry, second
seed, later rung, checkpoint reuse, Shared-JEPA training, selection,
calibration, G2, navigation, held-out, production, promotion, or deployment
authority.

## Restart order

Continue only in this order:

1. An eligible different agent independently rehashes both V15 authority
   documents, the V14 commit/review/terminal evidence, every V15 source and
   proof, and this handoff; reproduces the clarified V14 replacement and all
   CPU suites; and writes the canonical V15 review last as `PASS` or `BLOCK`.
2. If and only if the review is `PASS`, freeze the unchanged reviewed V15
   source/proof/handoff closure in a Git commit and record that commit. Any
   byte change invalidates the review and requires a new review.
3. With separate explicit user approval for out-of-sandbox/device-enabled
   access, run only the standalone visibility diagnostic against the
   caller-bound V15 review hash. Record the fixed receipt's file and content
   SHA-256. Only `pass_exactly_one_r9700` may continue. A failure grants no
   permission to alter the experiment; cleanup or repetition of the external
   receipt requires separate explicit approval.
4. With a fresh passing receipt and separate explicit user approval for the
   exact unsandboxed/device-authorized launch, invoke
   `scripts/execute_go2_observable_camera_ray_fit_v4_n5_gate_aligned_raster_nll_v15.py`
   explicitly with Python `-I -B`, the caller-bound review file hash, receipt
   file hash, receipt content hash, and at most five RGB workers. Do not start
   if any `.generated` mutator is active or the V15 root is no longer fresh.
5. Preserve the resulting V15 lifecycle bytes under every outcome. Only a
   complete unchanged 26-check `PASS` can justify a new source-free later-rung
   design and independent review; it does not itself authorize navigation or
   held-out maze tests.

This handoff is not an execution receipt. Held-out maze access remains behind
the reviewed V15 exact gate and the later navigation/training gates.
