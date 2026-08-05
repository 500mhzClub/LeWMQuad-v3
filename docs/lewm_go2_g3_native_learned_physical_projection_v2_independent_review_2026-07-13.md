# Go2 G3 native learned physical projection V2 independent review

Date: 2026-07-13

Verdict: **BLOCK**

The additive V2 candidate closes both frozen V1 failures through its advertised
public methods, and its supplied regression suite is green. It is nevertheless
not an authority boundary: the V2 instance retains a directly reachable V1
adapter whose public methods bypass the V2 digest and retraction-reservation
checks. The independent reproducer used that path to accept a post-commit
mutated and rehashed target and successfully remove its active learned evidence.

No downstream integration, G3 learned-evidence admission, production binding,
hardware execution, promotion, or navigation use is authorized from V2.

## Frozen artifacts reviewed

| Artifact | SHA-256 |
|---|---|
| `lewm/planning/native_learned_physical_projection_v2.py` | `327f3f7ab42ae39b416d54936bba6d39febdf6d85cea46c6acd7075c79716f40` |
| `lewm/tests/test_native_learned_physical_projection_v2.py` | `691e9d8a101044cb4b189f10a272bc5c633bf408724c657d66825c86651ca25b` |
| `docs/lewm_go2_g3_native_learned_physical_projection_v2_handoff_2026-07-13.md` | `83112bcf41b0a8c126aa22a69216c276406a1e27be0cf582761de977e37d993f` |
| `lewm/tests/test_native_learned_physical_projection_v2_independent_review.py` | `f979708cb9fcf9c6aaf1d8b4506b482eb0a48f84ebcae0764295e98db930b701` |

The candidate implementation, candidate tests, and handoff exactly match the
submitted hashes. This review did not edit them. The separate adversarial test
file is the only source artifact added by the reviewer.

## Blocking finding

### Reachable composed V1 engine bypasses every V2 retraction guarantee

V2 stores a complete `NativeLearnedPhysicalProjectionAdapterV1` instance in
its `__inner` slot at source lines 82 and 104. Name mangling is not an access or
authority boundary in Python: the exact object is returned by
`object.__getattribute__(adapter,
"_NativeLearnedPhysicalProjectionAdapterV2__inner")`. It retains the full
public V1 `issue_retraction()` and `commit()` surface.

The independent exploit performs this complete sequence using only synthetic
fixtures:

1. Issue and commit a valid projection through the public V2 adapter.
2. Mutate its committed pose after commit and recompute the nested admission and
   outer unkeyed content hashes, while leaving the original V2 issuance digest
   different.
3. Confirm public `adapter.issue_retraction()` rejects with `issued content`.
4. Obtain the composed V1 object from the V2 instance.
5. Call that object's V1 `issue_retraction()` and `commit()` methods on the same
   mutated target and current snapshot.
6. Observe a successful retraction receipt and the target disappear from
   `memory.learned_observation_ids`.

The concrete unauthorized receipt in the frozen run had transaction SHA-256
`8626943432c1e17992d58c81d402c979c4cb9207f14c47b88e6524561836582d`.
That receipt value is test-fixture evidence, not a registered identity.

V2 lines 197-267 correctly compare the public target and retraction packages
with their retained original issuance digests. Lines 306-389 correctly manage
LIVE, STALE, and CONSUMED reservations on the public V2 path. Those checks do
not protect a second reachable adapter. The composed V1 engine still contains
the exact frozen V1 issue-time omission and can commit the retraction directly,
without creating or consuming any V2 reservation record.

This is not merely base-class dispatch: V2 is correctly not a V1 subclass.
Composition moved the unsafe public adapter behind a mangled attribute but did
not eliminate it. The handoff statement that copied, forged, or mutated
retractions cannot enter the private V1 commit engine is therefore false for an
ordinary in-process caller.

## Public V2 checks that passed

The other seven independent adversarial probes pass:

- original target issuance digest is checked both before retraction issue and
  again before public V2 commit;
- copied, pickled/reloaded, `object.__new__`-forged, transferred, and replayed
  retraction packages cannot remove the target through V2;
- stale commit failure releases the target for an exact current-snapshot retry;
- proactive stale detection permits a replacement while the old package remains
  terminal across later snapshots;
- exactly one LIVE retraction is allowed per exact active target;
- successful target removal advances one revision, removes exactly that target,
  and preserves another active learned-observation identity;
- the V2 adapter contract is distinct from V1, is bound into issued admissions,
  and every exposed authority surface remains development-only, hardware false,
  and promotion false;
- adapter/package copy and adapter serialization authority remain denied;
- all V2 production runner, checkpoint, G2, calibration, and adapter constants
  remain `None`, and the production accessor fails closed.

These results show that the V2 state machine itself fixes the two frozen V1
defects. They cannot offset a reachable path around that state machine.

## Frozen V1 history preserved

Every frozen V1 byte and verdict remained unchanged:

| Artifact | SHA-256 |
|---|---|
| V1 implementation | `f8b149c685a4320ae938ff367edcf833047016250caae7699cddfe8026cc0634` |
| V1 candidate tests | `1f47ee15e46be1e8d5407ffa6f39f753b2dba92d15be67af8217ab4e146b5661` |
| V1 handoff | `caccd6204e394bd07e7c1f3d15b35775de20ac6fa2e17027d63efc5c326dbb2a` |
| V1 independent adversarial tests | `787b6d1ba10f24161ad355aef13a84e9891556d42d40693a02c803779b342ac3` |
| V1 independent BLOCK review | `5a41793bec15ea72ba89d5ce35e07746c44f3526dc4f16ce4f68a3ca30c9d07e` |

The frozen V1 independent suite still reports exactly `1 passed, 2 failed`:
post-commit mutation is accepted and a stale retraction strands active evidence.
V2 does not rewrite or supersede that negative evidence.

## Adjacent bytes checked

| Artifact | SHA-256 |
|---|---|
| revisioned physical memory | `13fccc662784c0a7eed75965a9d4154369666f26e804173482b461c55b8b9add` |
| revisioned physical memory tests | `a60c6f21cb0e40966216428c938e82024eafcaded95a874c53b542befb9065d4` |
| two-resolution configuration projection V2 | `3c858a89170f78a73f401c9534e231f24d6d91bb0469ea95eb00002158146107` |
| two-resolution configuration projection V2 tests | `8e61d29762cac2095d29c5e6341d63cac803c5f118a3eff7e8525b44b4985a3c` |
| G4 two-resolution frontier/viewpoint V2 | `5c84e79e558f51b75b00cf2baa26d7860302d6e3912ac14432dfc010efdc4f82` |
| G4 two-resolution frontier/viewpoint V2 tests | `c50e0d26be068228fe33530d3b2fa42b7520d20d93a0f6e7dc35a6c567ef963e` |
| G3 exact-equivalence V2 tests | `4069582829eedaf45b582003cbbdf517bbc8e3ab9a3370fd22abe16544bf4cf6` |
| legacy frontier/viewpoint source | `2ef20e8213a384e0f514705ca14c058eb7fbd81dcc4f6a53407414c1ba79e08e` |
| legacy frontier/viewpoint tests | `02d5a0b0459f6fde43e046b2b9f86d13d21e7392119b57626f0a398ce4c5241e` |

## Verification

All commands disabled external pytest plugins, capped OMP, OpenBLAS, MKL, and
NumExpr to one thread per process, and hid HIP, CUDA, and ROCr devices. The
independent V2, frozen V1, and adjacent groups were run CPU-only.

```text
V2 supplied candidate suite                         12 passed in 38.54s
V1 frozen candidate suite                           34 passed in 65.12s
memory + configuration projection suites            46 passed in 39.71s
G3 exact-equivalence + G4 V2 + legacy G4 suites     24 passed in 73.26s
                                                     -------------------
green candidate and adjacent regression total      116 passed

V2 independent adversarial suite                     7 passed, 1 failed
  failing exploit: reachable V1 engine removed the mutated active target

V1 frozen independent BLOCK suite                    1 passed, 2 failed
  expected frozen findings reproduced unchanged

py_compile: passed
git diff --check: passed for reviewed and independent V2 files
line-length check: passed (no independent-test line over 100 columns)
```

The implementation contains no Torch, NumPy, accelerator, file-opening,
checkpoint-loading, held-out, runtime, hardware, or production-input access
surface. The independent test opened no checkpoint, G2 report, real frame,
held-out scene, accelerator, hardware, or navigation input.

## Required successor closure

An additive V3 must not subclass, compose, retain, return, or otherwise expose
any `NativeLearnedPhysicalProjectionAdapterV1` or V2 adapter object. It may
reuse frozen pure geometry functions and immutable data types, but it must own
differently named issuance, digest, snapshot, committed-target, reservation,
and consumption state and execute a single commit path itself.

The successor review must prove all of the following:

1. no reachable composed/base adapter and no bound or unbound V1/V2 adapter
   method can issue or commit a V3 package;
2. exact original target and retraction digests are checked at issue and at the
   final pre-removal boundary;
3. stale and failed-stale reservations release for retry, while old packages
   remain terminal and exactly one current LIVE reservation exists;
4. copied, forged, rehashed, transferred, reloaded, replayed, and resurrected
   packages cannot enter memory;
5. target removal is atomic and bound to the exact active-memory identity;
6. V1, V1 negative evidence, V2 negative evidence, memory, projection, G3, and
   G4 adjacent suites remain unchanged and reproduce;
7. a different agent publishes a source-level PASS before any downstream use.

V2 remains a useful negative design record. It is not an integration candidate.
