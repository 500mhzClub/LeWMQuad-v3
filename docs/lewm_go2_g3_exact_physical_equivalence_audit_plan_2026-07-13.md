# Go2 G3 exact-physical equivalence audit plan

Date: 2026-07-13

Status: **candidate source only; 24-scene output not yet run or authorized**

## Purpose

The exact-map control must show that the new two-layer physical/configuration
memory is both correctly implemented and still capable of solving the task. A
single equality number conflates three different requirements, so the candidate
audit reports them separately without changing the preregistered gate:

1. **Discrete implementation equivalence.** Full physical cell squares are
   rasterized against world bounds and every rotated collision box. An
   independent array implementation then applies the frozen 89-cell FREE and
   69-cell OCCUPIED kernels with OCCUPIED precedence. The memory snapshot,
   spawn component, and selected A* distances must agree exactly.
2. **Analytic safety dominance.** Every configuration-FREE cell admitted by
   memory must have at least 0.47 m analytic point-to-rotated-box clearance and
   remain inside the world boundary. Unsafe FREE count must be zero in every
   development scene.
3. **Task usability.** The conservative spawn component must retain at least
   one physically line-of-sight-valid claim endpoint for every one of the 96
   development beacons. This prevents an all-UNKNOWN map from passing safety.

The historical strict binary-grid equality is reported independently. The
candidate result cannot promote itself if that legacy condition fails. A dated,
independently reviewed contract amendment would be required before learned G3
output.

## Centre-sampling rejection

An uncommitted one-scene development diagnostic confirmed why the physical
source cannot label only grid centres. On
`go2_deployment_medium_maze_09a7e5352fb3`, centre-sampled zero-inflation labels
produced 66 configuration-FREE cells below analytic 0.47 m clearance. The
full-square SAT raster produced zero unsafe FREE cells, retained 4/4 physical
claim endpoints, and conservatively withheld 1,922 cells that the historical
binary grid called FREE. A later captured-source smoke using the independent
analytic world-bound reference reproduced zero unsafe FREE, 8/8 A* distances,
and 4/4 endpoints while withholding 1,365 analytic-free cells. These figures
are design evidence only, not a gate result.

## Candidate implementation

- core:
  `lewm/benchmarks/go2_g3_exact_physical_equivalence.py`, SHA-256
  `b0155968a267afb08817987c3779e61e2e59b32e60281b1116a3757ac4fa461d`;
- captured runner:
  `lewm/benchmarks/go2_g3_exact_physical_equivalence_runner_v1.py`, SHA-256
  `4fbceaa49519d811de3f1508c99099c8b1ddda8cb7dacefcd8aa153a05f4a3b3`;
- isolated launcher:
  `scripts/audit_go2_g3_exact_physical_equivalence.py`, SHA-256
  `c22091aed4a554d87f912d4aa98c92ef3c529e61a39f8b2b06e568e36a56af3b`;
- focused tests:
  `lewm/tests/test_go2_g3_exact_physical_equivalence.py`, SHA-256
  `cae6e129ed2f9a1e8b527642dec93f705dcf7e01721f1c487cc722100f9546ad`;
- captured-loader tests:
  `lewm/tests/test_audit_go2_g3_exact_physical_equivalence.py`, SHA-256
  `8385ce570f5ff4418529400fbeaf603d1332e49382b85e0f818c06422b258a95`.

The nine focused tests pass in 0.17 seconds with native numerical threads capped
at one. Eight deterministic spawn-to-component A* probes per scene must also
match independent four-connected shortest distances. The launcher verifies and
executes the fixed-hash runner under `python -I -s` in a fresh child. That runner
rehashes the fixed 24-scene development manifest, geometry contract, all
relevant source before and after execution, and every scene semantic manifest.
Five project dependencies execute from captured, SHA-bound bytes. All worker
counts, including one, cross a process boundary through the captured runner's
`fork` process pool. Each job and result binds the runner and captured source
graph SHA-256
`0ec6f7194fae94eecaecdf9a4d2500164275a3023d08e8115e31273d8ae43009`,
and the coordinator independently recomputes every receipt before summary. The
audit is CPU-only and caps at six worker processes.

Execution remains pending the successor to the independently blocked
executor/reset authority, its different-agent review, and a review of this
audit source, so the output binds final source identities rather than the
rejected intermediate. No RGB, model, checkpoint, GPU, G2, held-out, or sealed
artifact was accessed.
