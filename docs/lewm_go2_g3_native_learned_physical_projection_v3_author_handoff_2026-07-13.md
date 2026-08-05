# Go2 G3 native learned physical projection V3 author handoff

Date: 2026-07-13

Status: **author candidate; development-only; awaiting independent review**

This document records implementation and test evidence from the authoring task.
It is not an independent source certification and does not authorize downstream
integration.

## Purpose

V3 is an additive standalone successor to the frozen V2 BLOCK. V2 fixed the V1
retraction defects on its public path but retained a reachable V1 adapter whose
public methods bypassed every V2 reservation and live-digest check.

V3 owns one complete adapter authority. It does not import, subclass, compose,
retain, or return a V1 or V2 adapter. It issues V3-specific admissions and
packages, reconstructs the hidden physical transaction itself, and calls memory
through one V3 commit path.

## Candidate artifacts

- implementation:
  `lewm/planning/native_learned_physical_projection_v3.py`
  - SHA-256:
    `c472b4792279a20fd7085189ea53d3a6c7d2c33343d86cc9063c73eea42f136f`
- focused author tests:
  `lewm/tests/test_native_learned_physical_projection_v3.py`
  - SHA-256:
    `d5113b9c98ad88f42315ce326cc8bb2b12933b3fc37471419282886f32f19129`

These hashes identify the candidate bytes before this handoff file was added.

## Standalone authority design

The V3 implementation imports only the frozen V1 calibration, raw outcome and
runner contract, immutable projection receipt, exceptions, and pure geometry or
hash helpers. No older adapter class is imported into the module.

`NativeLearnedPhysicalProjectionAdapterV3` owns distinctly named V3 state for:

- memory, projection, raw runner, calibration, and exact expected identities;
- the V3 adapter contract and unexported issuance capability;
- exact issued-package objects and their immutable original digests;
- exact issued snapshots and their immutable original digests;
- consumed packages and exact committed projection targets;
- exact retraction-package reservations and one LIVE reservation per target;
- a V3-only retraction observation sequence.

The V3 package type is
`QualifiedLearnedPhysicalDevelopmentTransactionV3`. V1 and V2 methods require
their own exact package types and cannot accept it. V3 deliberately exposes no
legacy state names used by unbound V1 methods and no composed V2 inner field.

Focused tests exercise both directions:

- bound V1/V2 adapters reject V3 projection and retraction packages;
- unbound V1/V2 `issue`, `issue_retraction`, and `commit` methods fail before
  reaching V3 state;
- direct V3 referents contain no V1/V2 adapter instance;
- the exact V2 mangled-inner exploit attribute does not exist.

## Projection and commit

V3 independently validates the exact current two-resolution snapshot, exact
runner-issued raw outcome, checkpoint/G2/calibration/source identities, both
map frames, `2:1` shapes, revisions, pose/camera identity, source geometry, and
covariance envelope.

The frozen geometry is preserved:

- FREE requires complete closed destination-square coverage for every frozen
  uncertainty transform;
- OCCUPIED is the closed union supercover of selected ordered-ray hit locations
  over all transforms;
- OCCUPIED precedes FREE and remaining observed support is UNKNOWN;
- native `0.05 m` source geometry is mandatory; `0.10 m` derivation or
  upsampling rejects.

V3 creates its own admission and package, retains their original content
digests, and consumes the raw outcome exactly once. Commit requires the exact
live package object, original digest, original snapshot digest, current
snapshot/revisions, V3 contract, map frames, memory configuration, and explicit
false authority flags.

The ordinary `PhysicalEvidenceTransaction` is never attached to or returned by
the package. V3 rebuilds it immediately before memory mutation and requires its
digest to equal the admission-bound digest.

## Retraction lifecycle

At retraction issue, the target must be the exact active committed V3 projection
and its current content must equal the immutable original issuance digest.

Each reservation binds:

- exact target observation and package identity plus original target digest;
- exact retraction package identity plus original retraction digest;
- exact snapshot identity plus original snapshot digest;
- one of LIVE, STALE, or CONSUMED.

Only one LIVE reservation may exist for an exact target. If its snapshot is no
longer current, it becomes terminal STALE and releases the target slot for a
fresh exact issue. A stale commit failure performs the same terminal transition.
The old package is retained as STALE evidence and can never be revived or
accepted on a later snapshot.

Immediately before target removal, V3 rechecks both original package digests,
the exact active committed-target registry, and the active learned-observation
identity. A successful atomic memory transaction removes exactly the target,
marks the retraction CONSUMED, and preserves ordinary replay rejection.

## Authority surface

Adapter, admission, receipt, and package surfaces all report and hash:

```text
development_only=true
hardware_execution_authorized=false
production_promotion_authorized=false
```

All V3 production runner, checkpoint, G2, calibration, and adapter constants
remain `None`; the production accessor fails closed. Adapter copy and
serialization authority are denied. Package copies, reloads, forgeries,
transfers, mutations, stale resurrection, and replays cannot enter memory.

## Author verification

Every command disabled external pytest plugins, set OMP/OpenBLAS/MKL/NumExpr to
one thread per process, and hid HIP, CUDA, and ROCr devices. Independent groups
were run concurrently on CPU.

```text
V3 focused author suite                              12 passed in 58.04s
V2 candidate + frozen V2 review probes               19 passed, 1 failed
V1 candidate + frozen V1 review probes               35 passed, 2 failed
revisioned memory + configuration projection         46 passed in 40.27s
G3 exact equivalence + G4 V2 + legacy G4             24 passed in 74.55s
```

The three failures are the exact frozen historical BLOCK findings:

- V2 reachable composed-V1-engine bypass: one expected failure;
- V1 post-commit mutation acceptance: one expected failure;
- V1 stale retraction stranding: one expected failure.

They remain negative evidence and were not edited or hidden. V3's adapted
versions of those exploits pass.

The V3 suite also covers concurrent LIVE rejection, failed and proactive stale
retry, terminal old-package rejection across later snapshots, target and
retraction rehash mutation, copy/reload/`object.__new__`/transfer/replay,
atomic target removal with another identity preserved, FREE translation and
rotation, boundary OCCUPIED supercover, OCCUPIED precedence, covariance and
native-resolution rejection, contradiction-to-UNKNOWN, exact retraction
recovery, hidden transaction, and explicit false authority.

`py_compile`, `git diff --check`, line-length, older-adapter-name, and forbidden
runtime-input surface checks completed cleanly.

## Frozen evidence preserved

V2 evidence remains byte-exact:

| Artifact | SHA-256 |
|---|---|
| V2 implementation | `327f3f7ab42ae39b416d54936bba6d39febdf6d85cea46c6acd7075c79716f40` |
| V2 candidate tests | `691e9d8a101044cb4b189f10a272bc5c633bf408724c657d66825c86651ca25b` |
| V2 author handoff | `83112bcf41b0a8c126aa22a69216c276406a1e27be0cf582761de977e37d993f` |
| V2 independent tests | `f979708cb9fcf9c6aaf1d8b4506b482eb0a48f84ebcae0764295e98db930b701` |
| V2 independent BLOCK | `8d924714db329ea23023322702168777e4831c050a65c4844ec5135533f22d63` |

V1 evidence remains byte-exact:

| Artifact | SHA-256 |
|---|---|
| V1 implementation | `f8b149c685a4320ae938ff367edcf833047016250caae7699cddfe8026cc0634` |
| V1 candidate tests | `1f47ee15e46be1e8d5407ffa6f39f753b2dba92d15be67af8217ab4e146b5661` |
| V1 author handoff | `caccd6204e394bd07e7c1f3d15b35775de20ac6fa2e17027d63efc5c326dbb2a` |
| V1 independent tests | `787b6d1ba10f24161ad355aef13a84e9891556d42d40693a02c803779b342ac3` |
| V1 independent BLOCK | `5a41793bec15ea72ba89d5ce35e07746c44f3526dc4f16ce4f68a3ca30c9d07e` |

Adjacent read-only hashes also match the V2 independent review:

- revisioned memory source/tests:
  `13fccc662784c0a7eed75965a9d4154369666f26e804173482b461c55b8b9add` /
  `a60c6f21cb0e40966216428c938e82024eafcaded95a874c53b542befb9065d4`
- two-resolution projection source/tests:
  `3c858a89170f78a73f401c9534e231f24d6d91bb0469ea95eb00002158146107` /
  `8e61d29762cac2095d29c5e6341d63cac803c5f118a3eff7e8525b44b4985a3c`
- G3 exact-equivalence V2 tests:
  `4069582829eedaf45b582003cbbdf517bbc8e3ab9a3370fd22abe16544bf4cf6`
- G4 two-resolution frontier/viewpoint source/tests:
  `5c84e79e558f51b75b00cf2baa26d7860302d6e3912ac14432dfc010efdc4f82` /
  `c50e0d26be068228fe33530d3b2fa42b7520d20d93a0f6e7dc35a6c567ef963e`
- legacy frontier/viewpoint source/tests:
  `2ef20e8213a384e0f514705ca14c058eb7fbd81dcc4f6a53407414c1ba79e08e` /
  `02d5a0b0459f6fde43e046b2b9f86d13d21e7392119b57626f0a398ce4c5241e`

## Explicit exclusions and next gate

No real V4/V5 checkpoint, G2 report, held-out scene, runtime input,
accelerator, hardware, production input, or navigation input was opened.

V3 remains a synthetic development candidate. It does not claim real runner or
source isolation, real G2 calibration, view diversity, traversal correction,
cold-start authority, promotion, hardware readiness, or navigation readiness.

A different agent must review the exact V3 source and test bytes and publish a
source-level verdict. No downstream integration or real-artifact binding is
authorized from this author handoff alone.
