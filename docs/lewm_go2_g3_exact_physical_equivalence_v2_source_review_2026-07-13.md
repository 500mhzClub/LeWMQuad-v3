# G3 V2 two-resolution source review

Date: 2026-07-13

Status: **BLOCK; authoritative V2 audit remains unauthorized and unrun**

## Reviewed candidate

This independent review used
`lewm_go2_g3_two_resolution_v2_design_contract_2026-07-13.md` as the governing
preregistration. It reviewed the additive V2 projection, exact-control core,
captured runner, isolated launcher, candidate amendment, and focused tests. It
did not modify the implementation and did not run the 24-scene audit.

| Artifact | SHA-256 |
|---|---|
| `lewm/planning/two_resolution_configuration_projection_v2.py` | `b49ed5c43fb2c2fb62b49264cb4336f91cf370b7dd1779aa1ff9742d15d7787a` |
| `lewm/benchmarks/go2_g3_exact_physical_equivalence_v2.py` | `1bf07b061ce23be94cc71824a3374650d32a1b61d6e584c76208962e06240633` |
| `lewm/benchmarks/go2_g3_exact_physical_equivalence_runner_v2.py` | `18847f956d88fe1964a47ff5419ece730d1c06a711a5fb2edfbd14ce2a53f9c3` |
| `scripts/audit_go2_g3_exact_physical_equivalence_v2.py` | `ecbdb72460680d8170a8237d463a1763c5bfe2fa2a04bae3c8d38cfa035e666b` |
| `lewm/tests/test_two_resolution_configuration_projection_v2.py` | `20603f22ff429f6f4636624c9cb3685306433eba0bd88cd91beabf2c6cb0e28d` |
| `lewm/tests/test_go2_g3_exact_physical_equivalence_v2.py` | `b4317c340cb70af6c608eaed43769a51b1b05fe397500138088f39a14f24835a` |
| `lewm/tests/test_audit_go2_g3_exact_physical_equivalence_v2.py` | `486dfba8960705c81661e23a74c15e948109197b406bb1bebe58185a2332ccce` |
| candidate amendment | `17b85d4ea035fde6df12c694f2362ddb6ecd9d29a7aec7ecdfef13c8497c6584` |
| governing design contract | `a82de141575efe9e12f0deea05477f558439d87bcb1af3bc36e0d377a36c95b1` |

The immutable V1 result remains present at SHA-256
`b7176cca80306768c6c851c61c2ba31636093b15bae777b1966cb2d56edc3d4c`.
No V2 candidate output exists.

## Blocking findings

1. **The implementation does not use the preregistered support and profile
   identities.** Exact rational recomputation reproduced the governing FREE,
   OCCUPIED, and projection hashes as `6fa138060d3df820d646e241fbc2786daf69ded355682afc8d566467f07acb4e`,
   `a18c0872c50ba749ae6737b0eeba428f5421b0d1518bd08251532f24ad6cb42c`,
   and `2b00cbe295ef4d0ef9f66e42b1aa7188751045240cba923392d83fd1bc709314`.
   The candidate changes canonical field names and adds fields, then freezes
   `94ec34b14f6d84383b50b5441993c627f35cc53b7ebaf6b739976233106afa62`,
   `afacfeca1c86a10ed9f2e7d31d49b87e8a510ff674a7f04421e39a3749ad0984`,
   and `190280ccf33a9de3a67fd8c7e23b1916a211b5f6341e0934979f56417f11aab0`
   instead. The candidate amendment repeats the replacement identities, so it
   contradicts rather than implements the preregistration.

2. **The required second-lattice identity and revision do not exist.** The V2
   snapshot carries the physical frame hash, physical revision, and physical
   content hash, but only a raw configuration origin and shape. It has no
   distinct configuration-map-frame hash, configuration revision, memory
   config hash, issued-snapshot identity, or projection-source identity. The
   per-scene serialized result drops even the physical snapshot identities.
   Consequently a public snapshot constructor accepted arbitrary non-hashes,
   physical revision `-7`, a `(NaN, Inf)` origin, and a non-boolean taint, and
   the planner routed over it. A real revision-1 V2 snapshot also remained
   accepted after its physical memory advanced to revision 2.

3. **The audit's independent projection is not independent and support mutation
   is not rejected.** Both projection paths consume the same mutable
   `FIXED_PROFILE_V2` offset tuples. Removing required FREE offset `(10,3)` via
   adversarial mutation changed a controlled cell from UNKNOWN to FREE while
   the profile content hash stayed unchanged; neither projection nor planner
   raised. This also means the candidate cannot prove production/independent
   316/276 equality under a support defect. The focused tests compare candidate
   constants to themselves rather than to the preregistered canonical records.

4. **The governing preregistration is absent from the captured source graph.**
   The runner binds the contradictory candidate amendment, but its
   `SOURCE_PATHS` does not include the two-resolution design contract. A result
   therefore could not prove which frozen 316/276 identity it was meant to
   implement.

5. **The imported launcher is source-substitutable.** Replacing the launcher's
   mutable expected hash and source-reader global caused its public
   `sealed_bootstrap_probe()` to execute reviewer-supplied bytes in the isolated
   child. Direct disk mutation is correctly rejected, but the importable helper
   does not meet the repository's one-shot execution-authority rule. The
   authoritative operation must perform its canonical no-follow read and fixed
   hash check in a non-returning CLI path that cannot accept or expose a
   caller-replaceable loader.

## Checks that passed

- Exact rational supports equal the candidate offset lists: 316 FREE and 276
  OCCUPIED, lexicographically sorted, strict subset relation, reflection under
  `k -> 1-k`, and extents `-9..10` / `-8..9`.
- All frozen alignment vectors passed: FREE includes `(10,3)` and excludes
  `(10,4)`; OCCUPIED includes `(9,4)` and excludes `(9,5)`; `(10,0)` is
  FREE-only.
- A separately written exact projection oracle matched 556 randomized
  configuration cells with zero mismatches. Out-of-domain support is OCCUPIED
  and OCCUPIED precedence is correct.
- V1 closed-square/rotated-box, exact LOS, component, and deterministic A*
  checks remain composed into V2. The current V4 source and contract bindings
  match their frozen hashes.
- The captured bootstrap executed the expected runner in a distinct worker
  process with all four native thread caps at one. Wrong runner hashes and
  mutated captured dependency bytes were rejected.
- 71 focused and adjacent tests passed in 7.19 seconds under CPU thread caps;
  the four candidate Python files also compiled. Static source inspection found
  no RGB, checkpoint, learned output, held-out, sealed-role, Torch, or GPU input
  path in the exact-control closure.

## Required remediation

Keep V1 and its result untouched. Before another independent review:

1. use the exact preregistered kernel records and `2b00...` projection core;
   extended V4/planning metadata may be bound in a separate envelope;
2. introduce explicit physical and configuration frame identities, a
   configuration revision, issued snapshot/content binding, strict typed
   validation, serialization, and revision-current planner checks;
3. independently rederive supports in the audit path and assert integrity at
   every projection/planning boundary, including mutation tests;
4. carry both lattice identities, revisions, shapes, sizes, supports, physical
   evidence content, and projection source through each scene record and final
   result;
5. bind this governing design contract in the captured source graph and close
   the importable source-substitution path;
6. add the preregistered wrong-alignment, wrong-frame, stale-revision,
   transaction/retraction/reset/serialization, complete-raster, and authority
   regressions.

Only a different-agent PASS over the remediated identities may authorize the
fixed 24-scene V2 audit.

## Remediated candidate independent rereview

Status: **BLOCK; authoritative 24-scene V2 audit is not licensed**

The remediated candidate was reviewed independently under new source
identities. No implementation file was edited and the authoritative audit was
not run.

| Artifact | Remediated SHA-256 |
|---|---|
| `lewm/planning/two_resolution_configuration_projection_v2.py` | `bab2a626a5c64f555a691c863ddb48e08aba474cf418eb647a3e3a44380a3be8` |
| `lewm/benchmarks/go2_g3_exact_physical_equivalence_v2.py` | `8c474b68d4fa63d9b194dfebcb75a0c204eaa074624a5f5d549bf4fa7c53133f` |
| `lewm/benchmarks/go2_g3_exact_physical_equivalence_runner_v2.py` | `2f9bdd5f79a49dab7522a0d73b374b2fcabd0e3876495b48900d2930f12a4cd7` |
| `scripts/audit_go2_g3_exact_physical_equivalence_v2.py` | `5847d6a483aa8fd4a5e1b4d23052e9c4d9cb2a7e5e5ac81800c23f6af57d44b1` |
| projection tests | `fd0f0bb039c54af01fb2cf59221413f50b0653a76c672a59a7897ed660d748c8` |
| exact-control tests | `765539571542c9f61accf2a8ca47a5741ebda94fe5b6b7a35e3a7f869260f42c` |
| launcher tests | `15e53a08d7401d6cf4b830bc849060296ed1fe0dbfbd04f2d6a49eaf71e76619` |
| candidate amendment | `51389dd7d8fb4b1ded454b646b19c60dbe9d2c895bbe6dfd30718ebd1f9d09a2` |

### Prior findings closed

- Independent exact-rational recomputation now reproduces the frozen 316/276
  lists and the exact `6fa138...`, `a18c08...`, and `2b00cb...` identities.
  The production and Fraction-derived audit implementations use separate
  support derivations, and production support mutation is rejected.
- Snapshots and per-scene records now carry distinct physical/configuration
  frames, shapes, sizes, revisions, content hashes, support hashes, memory
  configuration identity, and projection-source identity. Forged fields,
  changed origins/ratios, mutated snapshots, and stale physical/configuration
  revisions are rejected.
- The governing design hash `a82de141...` is captured in the runner source
  graph. The launcher is now CLI-only: importing it exposes no runner, loader,
  path, hash, callback, or execution helper, while the fixed CLI probe crosses
  an isolated child and worker boundary under all four thread caps.
- Full closed-square physical rasterization, translated/rotated contact, exact
  LOS, component equality, deterministic A*, V4 source binding, exact-only
  evidence authority, origin/reset/retraction behavior, and no restricted
  held-out/RGB/checkpoint/GPU input path remain established.

### Remaining blockers

1. **Final publication can overwrite a concurrently created result.** The
   launcher and runner check `output.exists()` before the long evaluation, but
   `_write_atomic()` ends with `os.replace(temporary, path)`. An adversarial
   reproduction passed a pre-existing sentinel path directly to this fixed
   runner function; the sentinel was replaced by the new JSON. The result is
   therefore neither exclusive-create nor one-shot under a check/write race,
   contrary to the amendment's refusal-to-replace clause. Publication must use
   an atomic no-replace primitive or reserve the canonical output exclusively
   before any scene work, with failure cleanup that cannot replace an existing
   candidate.

2. **Serialized and copied planning artifacts are accepted as live issuance.**
   `copy.copy(snapshot)`, `copy.deepcopy(snapshot)`, and
   `deserialize(snapshot.serialize())` each produced a distinct object that the
   original live planner accepted. Copied, deep-copied, and caller-reconstructed
   paths were also accepted, and a copied component passed integrity. Issued
   snapshots are tracked only by content hash, paths have no issuer-owned
   content/issuance identity, and the tests explicitly expect serialized replay
   to remain usable. This violates the governing execution-authority rule that
   serialization/copy/replay cannot become live evidence issuance.

3. **V2 drops execution-block evidence.** After an admitted CONTACT block was
   recorded at configuration cell `(15,16)`, the memory reported its execution
   block, but a new V2 projection still classified `(15,16)` as FREE. The V2
   projection consults only physical labels and omits the exact-centre block
   precedence retained by V1. This can route the robot straight back through a
   contact, stall, or veto location.

4. **The V2 planner has no bound frontier artifact or frontier operation.** It
   exposes connected components and A* only. There is no `frontier_cells`
   consumer that validates component issuance, snapshot/revision/support
   binding, or copied components. This leaves the online exploration interface
   required by downstream navigation incomplete and makes the component replay
   requirement untestable at its intended consumption boundary.

### Verification

The focused V2 suites plus adjacent V1 exact-control, revisioned-memory,
executor/reset, and V4 evidence suites passed `98/98` in `34.84 s` with
`OMP_NUM_THREADS=OPENBLAS_NUM_THREADS=MKL_NUM_THREADS=NUMEXPR_NUM_THREADS=1`
and both GPU visibility variables empty. The four candidate Python files
compiled. Passing tests do not close the uncovered blockers above.

V2 output remains absent. Immutable V1 source/runner/launcher/result hashes
remain respectively `b0155968...`, `4fbceaa4...`, `c22091ae...`, and
`b7176cca...`. A further remediated source identity requires another
independent review before the 24-scene V2 audit can be licensed.

## Final-block remediation implementation handoff

Status: **candidate source updated; not independently rereviewed; audit remains unauthorized**

The implementation author subsequently addressed the four remaining findings.
This entry records the handoff and is not an independent PASS:

- final publication uses verified directory descriptors and atomic hard-link
  creation, never replacement; pre-existing, concurrent-winner, symlink-parent,
  noncanonical-path, and temporary-file cleanup regressions pass;
- live planner authority is exact-object issuance for snapshots, components,
  frontiers, and paths; copy, deep-copy, serialization replay, and reconstructed
  values are rejected while the original artifacts remain usable;
- admitted CONTACT evidence maps each physical block cell to exactly one
  configuration centre with `//2`, OCCUPIED precedence, no second dilation,
  and a physical-revision-bound serialized receipt;
- deterministic `frontier_cells` production and validating consumption now bind
  the live snapshot, component, both revisions, frames, and support identities.

Candidate source identities submitted for the next independent review:

| Artifact | SHA-256 |
|---|---|
| two-resolution projection/planner | `3c858a89170f78a73f401c9534e231f24d6d91bb0469ea95eb00002158146107` |
| exact-control core | `a626a726b2837c6dd8cfacd6d7be3b796278b127ea998ff3a3b894bbf7d69823` |
| captured runner | `d759cb7fa395646d435bdd0af220a098d7d1e908970a30c4f17fc9e391c296e8` |
| one-shot launcher | `3f6fedf1614e01770fa080e870730da32864c65e5fc9e2bae12abdc52d79bad3` |
| projection tests | `8e61d29762cac2095d29c5e6341d63cac803c5f118a3eff7e8525b44b4985a3c` |
| exact-control tests | `4069582829eedaf45b582003cbbdf517bbc8e3ab9a3370fd22abe16544bf4cf6` |
| launcher/publication tests | `ddb055892b8a41ccc6402c0c9d846857fbf2fb989e2c72cca56a30c8eddce762` |

The focused suites passed `25/25`; the capped focused and adjacent closure
passed `132/132` in two CPU processes with GPU visibility disabled. Candidate
Python files compiled. The canonical `6fa138...`, `a18c08...`, and `2b00cb...`
identities and governing design hash remain unchanged. No V2 output was
created, and no authoritative audit was run.
