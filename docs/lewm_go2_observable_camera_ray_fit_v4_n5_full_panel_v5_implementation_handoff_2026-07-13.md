# V4 N5 full-panel V5 implementation handoff

Date: 2026-07-13

Implementation author: `/root/coordinator_v2_qa`

Status: **AUTHOR COMPLETE, DIFFERENT-AGENT REVIEW REQUIRED**

V5 is an additive remediation of the frozen V4 BLOCK. V1 through V4 source,
tests, handoffs, reviews, and BLOCK evidence were not edited. No canonical V5
PASS review JSON or experiment output was created. Exact optimization was not
run, and no dataset, RGB, model, checkpoint, protected-role, G2, held-out,
runtime, hardware, navigation, production, or promotion payload was opened.
All author verification was CPU-only with accelerators hidden.

## Parent V4 BLOCK

V5 binds the V4 handoff at file SHA-256
`4e0aa7e2efa266feb774a4b095cbddca105cfd046aac7a0da7f942f1b2b6925e`,
the independent review at
`7edeff73d6022a4086706907b03084ff080c9ad1d52ae91e8659fc6ecdc6b18c`,
the independent exploit test at
`2942b23215f506fa9893013d377f5bb4ce4b2327083a1806be4746bfdae56e9f`,
and the machine-readable BLOCK at file SHA-256
`d2224049a4ee2b793737802d06d91757c17d20b0457c1624517467638173c507`
and canonical content SHA-256
`0c34ec6931c8850a949498ca1b38f16548db76bc4d6e1e47994c6514898ff091`.
It also rehashes every retained V4 implementation and author-test artifact.

The V4 review established three remaining blockers:

1. the source reader opened the repository root as one absolute path, so a
   transient alias in an ancestor could be accepted before leaf reads;
2. success publication retained the claim directory descriptor but not the
   complete canonical output/claim ancestry, so replacing an output ancestor
   with an alias to the same tree was accepted; and
3. verification or finalization exceptions only closed the claim descriptor,
   leaving the sole completed attempt without a terminal failure receipt and
   leaving owned derived partials behind.

## Additive V5 artifacts

| Artifact | SHA-256 |
| --- | --- |
| `lewm/benchmarks/go2_observable_camera_ray_fit_v4_n5_full_panel_v5.py` | `cc28934be4fe1109feae3a31803e9e09502e968591268f80fc7124ba0a63f2c1` |
| `scripts/execute_go2_observable_camera_ray_fit_v4_n5_full_panel_v5.py` | `5dcc77a7434b64d3ae759b563b16db95e909bec9d1751dacc7657f6a740ac2e1` |
| `lewm/tests/n5_full_panel_v5_synthetic_execution.py` | `7601341cd92beb1a9a6738d2534e6f654a4058fe7d84b07547ac75f674fef608` |
| `lewm/tests/test_go2_observable_camera_ray_fit_v4_n5_full_panel_v5.py` | `80f51db295cad4d2a8494d1c61a1f605dac12cf558b5137d0eeee15611d88264` |

These hashes are the author freeze. Any source or test change requires new
hashes and a new review binding.

## Filesystem-root source walk

The V5 policy opens only the filesystem root by absolute path. It then walks
every component of the canonical repository root and requested relative
source path with descriptor-relative `O_DIRECTORY|O_NOFOLLOW` opens. Every
directory descriptor remains open until the leaf read completes.

Before, during, and after the leaf read, V5 compares full stable fingerprints
for the filesystem root, every named directory entry, every opened directory,
and the singly-linked regular leaf. Fingerprints bind device, inode, mode,
link count, owner, group, size, modification time, and change time. Transient
ancestor aliases, persistent aliases, component replacement, same-inode
metadata changes, leaf replacement, hard links, symlinks, and non-regular
leaves fail before evidence can be accepted.

## Retained output and claim ancestry

Before atomic claim, V5 opens and retains a component-wise no-follow descriptor
chain from the filesystem root through the canonical repository, output root,
seed parent, metric parent, and gate parent. It separately retains the open
claimed directory and its device/inode identity. The atomic staging rename,
parent fsync, later claim reads/writes, metric publication, and gate
publication are descriptor-relative.

Success stages require both the open claim identity and the complete retained
canonical chain to match their named entries and full fingerprints. Expected
directory metadata changes caused by V5 itself are refreshed explicitly and
only for the changed descriptor. Replacing or aliasing any output ancestor,
even back to the same tree, therefore prevents success publication. Every
claim and ancestry descriptor is closed on success, original failure, and
secondary terminalization failure paths.

V5 creates new `.n5.reservation-v5-*` private staging. Recovery recognizes
legacy and V2 through V4 staging namespaces, but resumes only a complete
staging record that rehashes and validates against the V5 reservation and
different-agent review contract. Other owned predecessor staging is recorded
and removed without claiming an attempt.

## Post-training terminalization

The end-to-end executor explicitly distinguishes `verification` and
`finalization`. Any exception after training calls terminalization before
descriptors close. Failure handling deliberately requires only the retained
claim descriptor identity, so it can finish even when canonical ancestry is
the cause of the failure.

Every created checkpoint, result, completion, metric receipt, and gate is
registered with its retained parent descriptor, role, full committed
fingerprint, and payload SHA-256. Terminalization removes an artifact only if
the named entry still matches the exact singly-linked regular fingerprint V5
created. Missing artifacts are recorded as absent; changed, replaced, linked,
or foreign artifacts are preserved and marked invalid. After cleanup, V5
writes and fsyncs `failed.json` through the owned claim descriptor, records the
failure stage and cleanup outcomes, and keeps retry authorization false. The
attempt directory remains terminal and cannot be reclaimed.

## Frozen execution contract

The production module still exposes no importable lifecycle function, class,
reservation, writer, or partial stage. All execution definitions remain
inside the canonical script's `__main__` branch. Frozen science remains seed
`20260710`, N=5, 400 full-panel updates, 2,000 frame exposures, four equally
weighted losses, final-update checkpoint selection, and matched plus cyclic
wrong-RGB evaluation. The schedule commitment remains
`62efec890e572623ab6d76e8c67337ee29badaf81638943ae56ed8da0a3a8634`.

## Author verification

Commands disabled external pytest plugins, capped OMP, MKL, OpenBLAS, and
NumExpr threads to one, and hid HIP, CUDA, ROCr, and HSA devices.

```text
V5 author/adversarial/source/lifecycle suite:       32 passed in 1.14s
V1-V5 retained author closure:                     111 passed in 2.16s
frozen V3 independent BLOCK reproducer:              7 passed, 8 failed
frozen V4 independent BLOCK reproducer:             14 passed, 3 failed
V5 isolated CPU contract smoke:                     PASS
py_compile for all four additive V5 artifacts:      PASS
```

The CPU smoke reproduced 400 updates, 2,000 exposures, full five-frame panels,
schedule SHA-256
`62efec890e572623ab6d76e8c67337ee29badaf81638943ae56ed8da0a3a8634`,
and frozen synthetic total loss `0.265`. It did not optimize a model or open
experiment inputs. The V5 passing counterparts cover all three frozen V4
failures plus source full-fingerprint mutation, descriptor-chain closure,
owned cleanup before terminal receipt, exact-owner removal, foreign/mutated
artifact preservation, verification/finalization stage recording, and durable
no-retry behavior.

## Required different-agent review

A reviewer other than `/root/coordinator_v2_qa` must:

1. rehash this handoff, all four V5 artifacts, every retained V1-V4 artifact,
   all four BLOCK records, independent exploit tests, and frozen numerical
   dependencies;
2. rerun the 32-test V5 suite, 111-test retained author closure, frozen V3 and
   V4 BLOCK reproducers, CPU contract smoke, and compilation with CPU threads
   capped and accelerators hidden;
3. independently prove that ordinary import exposes no lifecycle operation
   and that the single script entry closes every claim and ancestry descriptor
   on all success, failure, and secondary-failure branches;
4. attack the source walk before, during, and after each component open and
   leaf read using transient/restored aliases, replacement directories,
   symlinks, hard links, non-regular leaves, and same-inode metadata changes;
5. attack every canonical output ancestor and the claim entry at reservation,
   training publication, metric verification, and gate finalization, including
   aliases that point back to the original tree;
6. inject verification and finalization failures before and after each owned
   claim/derived publication; prove exact owned partials are removed, changed
   or foreign artifacts are preserved invalid, `failed.json` is written
   through the retained claim descriptor, and no retry is possible;
7. confirm the canonical V5 review JSON and canonical output root remain
   absent; and
8. only after every check passes, create the canonical different-agent review
   JSON from `expected_source_review_core(...)` and a separate reviewer report.

Until that review passes, exact execution fails closed. This handoff grants no
exact attempt, retry, N16, second seed, later-model training, G2, held-out,
selection, calibration, runtime, hardware, navigation, production, or
promotion authority.
