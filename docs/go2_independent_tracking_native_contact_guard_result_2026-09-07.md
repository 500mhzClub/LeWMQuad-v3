# Prospective contact-integrity guard completed; no native challenge run

The unfrozen independent tracking recorder now rejects a native error or changed
contact allocation before accepting each physics sample, and checks again at
termination. Array-size compliance alone can no longer admit a tape whose
simulator reports overflow. This closes a source-identified recording gap;
**no overflow in the old frozen recordings has been established**.

## Implementation and evidence

- The guard validates the effective CPU, single-environment, nondifferentiable
  collision configuration, float32/int32 dtypes, maximum 150 collision pairs and
  five contact slots per pair. The actual allocation is retained: a scene with
  fewer possible pairs is not misreported as having 750 contact slots.
- Seven exact installed Genesis source files are SHA-256-bound: package
  initialization, solver options, simulator stepping, rigid solver, collider,
  contact implementation and rigid-entity contact access. No native source is
  modified, recursively indexed, or exported.
- The new session wrapper checks the current step before the unchanged recorder
  accepts it. Failure latches; clearing a synthetic error or revisiting a record
  index cannot resume the attempt. Constructor failure destroys the owned scene.
- Terminal checking is separate from final snapshot/cleanup. A genuine physical
  stop remains recorded when native error state is clean. A later integrity
  failure preserves the stop evidence but classifies the tape as an
  infrastructure/recording failure, not a usable physical result.
- The result embeds initial/terminal capacity, attempted/passed sample counts,
  terminal status and first failure. Cohort admission checks this witness, and
  raw auditing checks its coverage against the actual recorded sample count.
  This metadata check does not itself reconstruct raw contact measurements or
  authorize native state as a command input.

The 37 new focused tests use synthetic native fields, including the actual
inherited recorder with synthetic robot getters. They cover same-step rejection
before getters, physical-stop retention, terminal failure, changed capacity,
missing/repeated indices, constructor cleanup, corrupted admission witnesses and
exact native-source identity. There is no Genesis initialization or checkpoint
access in those fixtures.

The adjacent regression completed **416 tests, zero failures/errors/skips**, in
294.817 seconds across 17 explicit files. Its durable report is
`.generated/navigation-development-staging.m6MDz1/independent_tracking_native_contact_guard_adjacent_v1.xml`,
SHA-256 `86042ad57b0eff2a93d154d5ec2049a7a6bfc7b19472c7a9399a5dc3f75c52cb`.
The process was observed terminal and the XML independently read during the
subsequent status turn. These are not 416 native trials or full-repository tests.

## Source identities and limits

Guard source:
`scripts/independent_tracking_native_contact_guard_development.py`,
SHA-256 `85a5d5caf2d0c71297d7711c7b87d8d2be64b27501ad35f4de00f0e39f654b8d`.
Session source:
`scripts/independent_tracking_session_development.py`,
SHA-256 `39eef5980432724ed6a3521713e6f710a0d9d6835fb1e0e1a5f6bfdd8f4cbdd1`.
Collection source:
`scripts/independent_tracking_collection_development.py`,
SHA-256 `888f9120946032132a4e718cd4c0723357d8899dfa56d617e4da62a6b6c39d75`.

The original 701-source tracking replay, 771-source learning supervisor and
786-source matched learning definition remain unchanged. The later memory
supervision work changes only the unfrozen challenge launcher/tests/protocol;
its verification must be reported separately from this completed 416-test run.

The challenge remains unlaunched and unqualified. Next: verify the new outside
memory supervisor, complete a genuine resource review, and preserve the original
twelve-layout collection and 36-fit study's scheduling priority. Independent
tracking, continued closed-loop execution, low-friction reliability, JEPA and
online-rollout benefits, memory/backtracking, novel-maze completion and hardware
evidence remain separate requirements of the ultimate goal.
