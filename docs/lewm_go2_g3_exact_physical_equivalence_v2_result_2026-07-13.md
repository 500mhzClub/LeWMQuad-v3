# G3 V2 exact-physical equivalence result

Date: 2026-07-13

Status: **PASS for the immutable exact-control V2 candidate; no learned or
production promotion is authorized**

## Reviewed artifact

The independent review treated
`.generated/go2_g3_exact_physical_equivalence/v2/candidate.json` as immutable.
It did not invoke the publisher, replace either candidate, or modify source,
scene data, checkpoints, readiness records, or generalization records.

| Identity | SHA-256 |
|---|---|
| V2 candidate file bytes | `22a15e0fa9085d23d48fbb09d3fc3c6e64618739b69951c38e9f0ee869a9fb5b` |
| V2 canonical candidate content | `92986eb0454fba20ed06669db8d2e4b91d8a8e40a6306533951ecd51dd13c1db` |
| V2 canonical pre-runner summary content | `52a34b084192fd53a8ebf24f4b8fcfccaf4562c185ed629a9d34be4c6ea3f96a` |
| Captured source graph | `972f6c2ac0c33278d3685da465c89fe8d5939e36b69e7ff3eda50257aa6bb561` |
| Projection contract | `2b00cbe295ef4d0ef9f66e42b1aa7188751045240cba923392d83fd1bc709314` |
| FREE support kernel, 316 offsets | `6fa138060d3df820d646e241fbc2786daf69ded355682afc8d566467f07acb4e` |
| OCCUPIED support kernel, 276 offsets | `a18c0872c50ba749ae6737b0eeba428f5421b0d1518bd08251532f24ad6cb42c` |
| Governing design | `a82de141575efe9e12f0deea05477f558439d87bcb1af3bc36e0d377a36c95b1` |

The JSON was parsed with duplicate-key and non-finite-number rejection. Its
bytes are exactly canonical sorted compact JSON plus one newline. Recomputing
SHA-256 after removing only the outer `content_sha256` field reproduced
`92986e...`; independently removing the runner envelope reproduced the sealed
summary hash `52a34b...`.

## Independent result checks

The serialized audit passed all of the following checks without trusting its
headline booleans:

- exactly 24 unique development scenes and 24 unique semantic manifest hashes;
- exactly 24 unique physical/configuration/snapshot/projection identity tuples;
- all 24 physical rasters complete, with zero UNKNOWN physical cells;
- exact shared lattice origins, 0.05 m physical cells, 0.10 m configuration
  cells, and an exact 2:1 physical shape on both axes in every scene;
- all 24 distinct physical and configuration map-frame hashes independently
  reproduced from their serialized identities;
- all 24 independent morphology comparisons have zero label mismatches;
- all 24 independent connected-component comparisons have zero cell
  mismatches and matching component sizes;
- all 192 deterministic four-connected A* probes agree, with zero route
  mismatches;
- all 96 of 96 beacon claim endpoints are retained, four in every scene;
- zero unsafe configuration-FREE cells across all scenes;
- all 24 execution-block receipts independently reproduce
  `730715bc1361be46bb74f934c20858e4af47bb8695038f4a007eaa12d11d776f`;
- all 24 canonical per-job hashes and all 24 canonical per-result hashes match
  their sealed job bindings; and
- all 15 frozen source bindings match the current file bytes, and their source
  graph independently reproduces `972f6c...`.

As a stronger result check, all 24 scenes were independently re-evaluated from
their frozen manifests and geometry contract. Six CPU worker processes were
used, each capped to one native thread with HIP and CUDA visibility empty.
Every regenerated full scene record matched the published canonical record,
and every regenerated result hash matched its sealed receipt. This rerun
reproduced 192 A* probes, 96 retained endpoints, zero unsafe FREE cells, zero
independent-label mismatches, zero component mismatches, and zero route
mismatches.

The result is intentionally conservative rather than strict binary equality.
It contains 13,260 conservative false-reject cells and 25,644 strict-binary
label mismatches, so `legacy_strict_binary_equivalence_pass` is correctly
false. These cells do not create unsafe FREE space and do not invalidate the
preregistered V2 conservative-equivalence gate.

## Frozen closure

| Bound file | SHA-256 |
|---|---|
| `config/go2_generalization_geometry_v2.json` | `e7d0627d1de259c6e01dabe142aa55e69fed3e75c9c745974d437d7682d40a52` |
| `config/go2_generalization_v4/development.json` | `563f240a023309af42a05a9a8f29008f02a0629dee9f77f03568f779d1166d41` |
| `docs/lewm_go2_g3_exact_physical_equivalence_v2_amendment_2026-07-13.md` | `7ab696ec97e864fe073b9d9cad70e403da22251dc7c2634cf8efff5da902b67c` |
| `docs/lewm_go2_g3_two_resolution_v2_design_contract_2026-07-13.md` | `a82de141575efe9e12f0deea05477f558439d87bcb1af3bc36e0d377a36c95b1` |
| `docs/lewm_go2_observable_camera_ray_evidence_v4_contract_2026-07-12.md` | `0a17cc94056ef5c53d2a96266cb21a5500eb3a9ea13e62f02f296b97455bcdee` |
| `lewm/benchmarks/go2_g3_exact_physical_equivalence.py` | `b0155968a267afb08817987c3779e61e2e59b32e60281b1116a3757ac4fa461d` |
| `lewm/benchmarks/go2_g3_exact_physical_equivalence_v2.py` | `a626a726b2837c6dd8cfacd6d7be3b796278b127ea998ff3a3b894bbf7d69823` |
| `lewm/benchmarks/go2_g3_exact_physical_equivalence_runner_v2.py` | `d759cb7fa395646d435bdd0af220a098d7d1e908970a30c4f17fc9e391c296e8` |
| `lewm/benchmarks/go2_observable_camera_ray_evidence_v4.py` | `708d368e461fe60aacb860dda5b0cbfd1acaf43e5cb3ae18a77bb48de739fb85` |
| `lewm/planning/geometry_contract.py` | `6873a9550399a5decc90e4a31b2945e54074bdb56855a035924f49b4511c813b` |
| `lewm/planning/revisioned_physical_configuration_memory.py` | `13fccc662784c0a7eed75965a9d4154369666f26e804173482b461c55b8b9add` |
| `lewm/planning/two_resolution_configuration_projection_v2.py` | `3c858a89170f78a73f401c9534e231f24d6d91bb0469ea95eb00002158146107` |
| `lewm/planning/zero_inflation_exact_physical_adapter_v1.py` | `2dc1629750a6487740187a1464c3d65f42d9fa78e491e8470a0f0cbfbf5cacad` |
| `lewm_worlds/lewm_worlds/manifest.py` | `5679768016226e89e385ec7a7238616416248a9a1194b898ecb9078662f6a888` |
| `scripts/audit_go2_g3_exact_physical_equivalence_v2.py` | `3f6fedf1614e01770fa080e870730da32864c65e5fc9e2bae12abdc52d79bad3` |

## V1 preservation and claim boundary

The immutable V1 candidate remains byte-for-byte unchanged:

| V1 identity | SHA-256 |
|---|---|
| V1 candidate file bytes | `b7176cca80306768c6c851c61c2ba31636093b15bae777b1966cb2d56edc3d4c` |
| V1 canonical candidate content | `070392510e976ca753414ab3881d8240654d152d9e8197b1f689c8c39c26f4aa` |
| V1 canonical pre-runner summary content | `d317e5c9e8e649cb7eef8808660498e0d148f5dabb84941abd9e7a9e210e10c2` |

This PASS establishes the exact-control, exact-physical two-resolution audit
result only. Every scene records exact-physical authority, one exact
observation, zero learned observations, and `production_promotion_authorized =
false`; the candidate also records `learned_projection_implemented = false`.
It therefore makes no learned-projection, held-out generalization, runtime
readiness, deployment safety, or production-promotion claim.
