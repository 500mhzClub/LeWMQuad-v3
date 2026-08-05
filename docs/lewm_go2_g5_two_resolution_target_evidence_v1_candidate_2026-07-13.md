# G5 two-resolution target evidence V1 candidate

Date: 2026-07-13

Status: synthetic-only additive candidate; not production authorized

## Scope

This unit closes one narrow part of the navigation integration gap. It binds
an exact live G3 V2 0.10 m configuration snapshot/component to runner-owned
V5 observations expressed on the 0.05 m physical lattice, then issues
immutable positive or negative G5 evidence on the 0.10 m lattice.

It does not change the passed reversible target posterior, the passed legacy
G5 evidence authority, the reviewed V5 lifecycle, or the reviewed G3 V2
projection. It does not load a model, checkpoint, dataset, held-out scene, GPU,
or simulator. All executable tests use synthetic issued V5 outcomes.

Additive files:

- `lewm/planning/two_resolution_target_evidence_v1.py`
- `lewm/tests/test_two_resolution_target_evidence_v1.py`
- this document

## Exact lattice rule

- Physical evidence and visibility are runner-owned 0.05 m cells.
- Candidate and posterior evidence cells are G3 configuration 0.10 m cells.
- The physical shape must be exactly twice the configuration shape on both
  axes.
- Physical and configuration frames must have exactly the same metric origin,
  distinct frame identities, and cell sizes 0.05 m and 0.10 m respectively.
- Conversion calls `physical_frame.cell_center` followed by
  `configuration_frame.world_to_cell`, and checks that the result is exactly
  `(physical_x // 2, physical_y // 2)`.
- The focused high-index test proves `(122, 86)` converts to `(61, 43)` and
  cannot survive as the same numeric cell.

Positive physical-cell mass is summed into its configuration parent. Negative
evidence is stricter: all four physical children of a configuration cell must
be runner-visible, runner-certified FREE, and have a detection probability.
The issued configuration probability is the minimum of those four values.
Partial visibility therefore cannot erase an entire 0.10 m hypothesis cell.

UNKNOWN, occupied, disconnected, and known target cells cannot enter the
candidate domain or issued evidence. Target physical cells are converted to
configuration cells and removed from the exact connected FREE component.

## Bound identities

Every context binds:

- exact live G3 V2 snapshot and exact live planner-issued component objects;
- physical and configuration frame records and hashes;
- physical and configuration shapes;
- physical and configuration revisions;
- physical content, projection source, snapshot, and component hashes;
- G3 profile, FREE support, and OCCUPIED support hashes;
- V5 runner wrapper, captured launcher, and captured core execution identity;
- evaluated checkpoint file hash;
- raw outcome file and canonical content hashes;
- camera calibration and pose provenance hashes;
- target ID, physical visibility/evidence cells, and candidate/target-exclusion
  configuration cells.

The exact source-issued V5 outcome is consumed once by context issuance. Each
context leases exactly one non-copyable writer. That writer issues exactly one
evidence record. The evidence record has an exact-object single-use consume
operation for the future posterior bridge. Clones, foreign objects, replay,
and stale G3 snapshots fail closed.

## Public candidate surface

Module: `lewm.planning.two_resolution_target_evidence_v1`

Synthetic V5 input:

```python
source = SyntheticV5TargetOutcomeIssuerV1(_synthetic_test_fixture=True)
outcome = source.issue(snapshot=..., outcome_kind=..., ...)
```

G3/V5 binding and issuance:

```python
issuer = TwoResolutionTargetEvidenceIssuerV1(
    projection=projection,
    planner=planner,
    outcome_source=source,
    runner_execution_identity=runner_identity,
    checkpoint_file_sha256=checkpoint_sha256,
    camera_calibration_sha256=calibration_sha256,
    _synthetic_test_fixture=True,
)
context = issuer.issue_context(snapshot, component, outcome)
writer = issuer.open_writer(context)
evidence = writer.issue_positive()  # or issue_negative()
issuer.consume_evidence(evidence)
```

Relevant output properties:

- `context.candidate_domain`: exact 0.10 m connected FREE cells after target
  exclusion.
- `context.excluded_target_configuration_cells`: converted target cells.
- `context.physical_cell_size_m`: 0.05.
- `context.posterior_cell_size_m`: 0.10.
- `positive.localized_distribution`: 0.10 m configuration-cell mass.
- `negative.visible_detection_probability`: conservative 0.10 m negative
  evidence.
- `evidence.posterior_cell_size_m`: 0.10.
- every candidate object exposes `production_eligible is False`.

This candidate deliberately does not issue a `TargetPosteriorSnapshot`. The
passed legacy posterior is tied to its passed single-resolution context issuer
and physical lattice. A later additive bridge or versioned posterior must
consume the exact evidence record without mutating those passed bytes, then
expose an exact current 0.10 m posterior to the router. Treating this evidence
candidate as a current posterior would be a contract violation.

## Production closure

All production bindings remain `None`:

- `PRODUCTION_G3_V2_SNAPSHOT_SOURCE`
- `PRODUCTION_G3_V2_COMPONENT_SOURCE`
- `PRODUCTION_V5_RUNNER_EXECUTION_IDENTITY`
- `PRODUCTION_V5_CHECKPOINT_FILE_SHA256`
- `PRODUCTION_V5_RAW_OUTCOME_SOURCE`
- `PRODUCTION_V5_CAMERA_CALIBRATION_SHA256`
- `PRODUCTION_G5_TWO_RESOLUTION_EVIDENCE_ISSUER`

`require_production_two_resolution_target_evidence_issuer()` raises before any
work. Production closure requires independently reviewed real V5 output and
authority identities plus a separately reviewed posterior bridge.

## Rejection coverage

Focused synthetic tests cover:

- positive sum aggregation and conservative four-child negative aggregation;
- high-index same-cell confusion;
- wrong live snapshot, component, frame, shared origin, 2:1 shape, physical
  revision, configuration revision, support hash, runner identity, checkpoint,
  calibration, and raw outcome identity;
- copied snapshot/component/outcome/context/evidence;
- replayed outcome/context writer/evidence;
- stale projection;
- partial visibility;
- UNKNOWN, target, and non-FREE configuration cells;
- positive/negative kind confusion;
- absent production identities and explicit synthetic-only construction;
- no accelerator or model-loading surface.

## Preserved byte identities

The following hashes were checked after the additive implementation:

- passed legacy posterior source:
  `b7f42f90accc9b44f9c38c386318e6775a26d3184d03086d14904487384f14f3`
- passed legacy posterior tests:
  `813ede3e46770b41d617ab90efb5e43ba77c4f99e411c44ce4638f2707cc90ce`
- passed legacy G5 runner-owned authority:
  `f7009462fc53e7c23adfe21fe8f6cd2d40b42753ab192536097812eb26e756a8`
- passed legacy G5 authority tests:
  `b3507fb837a3dc8f983cee8290a0e288b5a8e8d05ed999a9dbc5e79a4d6f6a98`
- reviewed V5 core:
  `62a19f3028e9152120af990528752431b996f56b4bc9b62db32eba47ae235a1f`
- reviewed V5 launcher:
  `7f273649fa6c8b4256c552359927fc20bb59d1bfbd5b47194a3f5a941c5b8958`
- reviewed V5 runner/finalizer/publisher wrappers:
  `37402f0f75a7a4f475539e269e77aeae072ce80b0af0bcb4147e2ec1b33ff57a`,
  `f0426201f5344d0eb1d43e183e4755ac8fd7aecdc9af6e5b7c19076af3f5dc34`,
  `4e045365dadb28bd37cdbb49808bef7528d4e5cb0c3e77ff5aae678559174fab`
- reviewed G3 V2 projection source:
  `3c858a89170f78a73f401c9534e231f24d6d91bb0469ea95eb00002158146107`

## Result

Final clean CPU-only results:

- focused two-resolution G5 evidence tests: 22 passed in 70.54 s;
- preserved legacy posterior plus G5 authority tests: 50 passed in 12.11 s;
- preserved G3 V2 projection tests: 14 passed in 37.65 s;
- total focused/compatibility coverage: 86 passed;
- `py_compile`, `compileall`, `pyflakes`, and whitespace checks: pass.

Additive file hashes:

- source:
  `f731b848f6b7ced3b07e11d4f9edca81daa8c66f083f9d503ed069809e38a9a2`
- focused tests:
  `e33dbf595fe27c18c2fddf89cc8f22a005574f67348c2d8746b8ee1ca039de26`

The documentation hash is reported externally after this result block is
sealed, avoiding a self-referential document hash.
