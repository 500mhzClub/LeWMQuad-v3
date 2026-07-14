# V4 N5 full-panel V5 independent review

Date: 2026-07-13

Reviewer: `/root/camera_v5_independent`

Verdict: **PASS**

The frozen additive V5 package closes all three findings in the frozen V4
BLOCK. The candidate files were not modified. The review created one distinct
reviewer test, this report, and the canonical machine-readable review record.
It did not run the exact experiment or open experiment data, RGB, a checkpoint,
protected roles, G2, held-out, runtime, hardware, navigation, production, or
promotion payloads. All dynamic review work was CPU-only in temporary paths
with accelerator visibility disabled.

## Frozen candidate

| Artifact | SHA-256 |
| --- | --- |
| `lewm/benchmarks/go2_observable_camera_ray_fit_v4_n5_full_panel_v5.py` | `cc28934be4fe1109feae3a31803e9e09502e968591268f80fc7124ba0a63f2c1` |
| `scripts/execute_go2_observable_camera_ray_fit_v4_n5_full_panel_v5.py` | `5dcc77a7434b64d3ae759b563b16db95e909bec9d1751dacc7657f6a740ac2e1` |
| `lewm/tests/n5_full_panel_v5_synthetic_execution.py` | `7601341cd92beb1a9a6738d2534e6f654a4058fe7d84b07547ac75f674fef608` |
| `lewm/tests/test_go2_observable_camera_ray_fit_v4_n5_full_panel_v5.py` | `80f51db295cad4d2a8494d1c61a1f605dac12cf558b5137d0eeee15611d88264` |
| author handoff | `df3d58eff6b582a113beb9d558c3e210f7a22acd38763f55037ae86609dc8b5c` |

The parent V4 BLOCK was independently rehashed at:

- review: `7edeff73d6022a4086706907b03084ff080c9ad1d52ae91e8659fc6ecdc6b18c`;
- exploit test: `2942b23215f506fa9893013d377f5bb4ce4b2327083a1806be4746bfdae56e9f`;
- BLOCK JSON file: `d2224049a4ee2b793737802d06d91757c17d20b0457c1624517467638173c507`;
- BLOCK canonical content: `0c34ec6931c8850a949498ca1b38f16548db76bc4d6e1e47994c6514898ff091`.

`preflight_static_authority()` also rehashed the preregistration, structural
trigger, terminal invalidation, retained V1-V4 evidence, all four BLOCK
records, and the frozen numerical source graph.

## Closure findings

### Filesystem-root source acquisition

PASS. The policy opens `/` once, then acquires every repository and requested
source component with descriptor-relative `O_DIRECTORY | O_NOFOLLOW`. It keeps
all directory descriptors live and compares complete device, inode, mode, link
count, owner, group, size, mtime, and ctime fingerprints before the leaf read,
after every component open, and after the read. The leaf is no-follow,
singly-linked, and regular.

Independent attacks confirmed that a moved ancestor temporarily restored
through a symlink is rejected before any `os.read`, same-inode directory
metadata changes are rejected after the read, symlink/hard-link/FIFO leaves are
rejected, and all source descriptors close on both success and injected read
failure.

### Canonical output and claim ancestry

PASS. Before atomic claim, the executor retains a no-follow descriptor chain
from `/` through the repository, output root, seed root, metric parent, and gate
parent. It separately retains the claimed staging directory descriptor before
the descriptor-relative rename. The seed parent is fsynced immediately and
the complete chain is identity- and fingerprint-checked at every success
stage.

The distinct reviewer suite compiled the nested frozen executor definitions
without invoking dispatch and exercised the actual production chain functions
against temporary paths. Replacing an output ancestor with a symlink back to
the same tree and replacing the canonical claim entry were both rejected.
Every retained claim and ancestry descriptor was proven closed.

### Post-training terminalization

PASS. `execute_exact` catches verification and finalization exceptions while
the claim descriptor is live, invokes `_terminate_failure`, and closes the
claim and ancestry descriptors in `finally`. Failure cleanup uses retained
parent descriptors and removes an artifact only when the current entry is a
singly-linked regular file with the exact recorded full fingerprint. It then
writes and fsyncs `failed.json` through the retained claim descriptor with
`retry_authorized: false`.

The actual nested executor was tested at four boundaries: before and after
metric publication, and before and after gate publication. Each case removed
the exact owned checkpoint, result, completion, and any derived partials;
published one terminal failure receipt with the correct stage; left the attempt
unreclaimable; and closed every descriptor. Separate checks proved that changed
owned artifacts and unregistered foreign artifacts are preserved as invalid,
exact owned artifacts are removed, terminalization still reaches the original
claim after its canonical entry is replaced, and descriptors close even when
terminal receipt creation itself raises a secondary exception.

## Import and science contract

PASS. Ordinary import defines no executor-owned function or class and exposes
no reservation, writer, partial stage, or execution entry. All lifecycle values
remain inside the sole `__main__` branch.

The frozen numerical contract remains identical to V1: seed `20260710`, N=5,
400 full-panel updates, 2,000 exposures, equal four-way loss weighting,
final-update checkpoint selection, matched plus cyclic wrong-RGB evaluation,
float32, and schedule SHA-256
`62efec890e572623ab6d76e8c67337ee29badaf81638943ae56ed8da0a3a8634`.
The isolated launcher retains `-I -B`, GPU0-only HIP visibility, removal of
`HSA_OVERRIDE_GFX_VERSION`, and one native thread per family. The Raphael iGPU
remains forbidden by the frozen experiment contract.

## Verification

Every pytest command disabled external plugins, set OMP, MKL, OpenBLAS, and
NumExpr threads to one, and hid HIP, CUDA, and ROCr devices. The project Python
used the system pytest module path; the isolated smoke used the project Python
directly with accelerator visibility empty.

```text
V5 author focused suite:                         32 passed in 1.07s
distinct V5 independent reviewer suite:          23 passed in 1.48s
retained V1-V5 author closure:                   111 passed in 2.16s
frozen V3 BLOCK reproducer:                        7 passed, 8 failed
frozen V4 BLOCK reproducer:                       14 passed, 3 failed
py_compile for candidate plus reviewer test:     PASS
isolated CPU contract smoke:                     PASS
```

The CPU smoke reproduced 400 updates, 2,000 exposures, full five-frame panels,
the frozen schedule hash, and synthetic total loss `0.265`. It did not perform
optimization or open experiment inputs.

## Review record and authority

The canonical review JSON is
`docs/lewm_go2_observable_camera_ray_fit_v4_n5_full_panel_v5_independent_review_2026-07-13.json`.
Its canonical content SHA-256 is
`441b0854fc50eda49b4124bd40d5e4beedaedfa41a4e99e1231b3ee81fa0d11d`
and its newline-terminated file SHA-256 is
`81345d133e53da1911d2561c6eaab74c341645fbf45dbefdf89bf730fed36cb0`.
It was constructed exactly from `expected_source_review_core(...)` with reviewer
`/root/camera_v5_independent` and only the two frozen successor source bindings.

This PASS authorizes one fresh exact N5 full-panel attempt and its bound metric
verification/finalization only. Retry, N16, a second seed, later-model V5
training, G2, held-out, selection, calibration change, runtime, hardware,
navigation, production, and promotion remain unauthorized. The canonical
experiment output root remained absent throughout review.
