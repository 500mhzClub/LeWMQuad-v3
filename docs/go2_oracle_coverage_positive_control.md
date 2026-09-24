# Go2 oracle coverage positive control

`lewm.benchmarks.go2_oracle_positive_control` is a privileged development-only
diagnostic. It does not evaluate JEPA, memory, learned heads, or perception. It
establishes whether the static scene geometry plus the repository's primitive
motion contract admit complete coverage and true beacon claims.

The control assumes:

- the exact manifest occupancy and fixed manifest spawn;
- 8-connected A* on the canonical 0.20 m inflated planning grid;
- the versioned geometry in `config/go2_generalization_geometry_v1.json`,
  including its oracle grid, configuration-space inflation, claim radius, and
  preferred standoff;
- conservative forward/rear/width swept probes as action-risk telemetry, not a
  second static collision hull layered over the configuration-space grid;
- five 0.1 s substeps from `config/go2_primitive_registry.yaml` per primitive;
- a true claim requires metric distance, unobstructed point-geometry line of
  sight, and heading agreement;
- coverage is normalized over the spawn-connected inflated free component;
- repeated beacon colors are report metadata; claim identity is the unique
  manifest object ID.

Run the existing development allow-list only:

```bash
.generated/venvs/genesis_render_vulkan/bin/python -m \
  lewm.benchmarks.go2_oracle_positive_control \
  --output .generated/oracle_positive_control/development/report.json
```

Run a versioned generalization development panel directly from its public
development manifest:

```bash
python3 -m lewm.benchmarks.go2_oracle_positive_control \
  --development-manifest config/go2_generalization_v3/development.json \
  --output \
    .generated/oracle_positive_control/generalization_v3_development/report.json
```

Manifest mode reads only `validation_scenes`, infers the matching materialized
development corpus, and verifies the geometry-contract hash, semantic manifest
hash, family, and beacon count for every requested scene. `--scene-id` may
select a subset, but it must name a validation scene in that manifest.

The CLI refuses scene-list and output paths labelled `sealed`, `final_eval`, or
`final_test`. A missing claim anchor is classified as `scene_geometry`; a
missing route to an existing connected anchor is `planner`; failure to execute
an existing route is `follower`; exhausting the horizon with continued progress
is `budget`.

`all_beacons_claimed` is reported separately from the strict positive-control
success gate. The latter requires every beacon plus 90% reachable-space
coverage within the fixed 2400-tick horizon. This distinction matters for the
large family: a scene may establish task reachability while honestly reporting
that near-total area coverage exceeded the benchmark horizon.

Every report records the geometry contract's canonical SHA-256 and source
artifact verification status. The contract is currently provisional pending
physical footprint calibration, which the report preserves rather than hiding.
Scene validity searches every spawn-connected cell inside the true 1.20 m claim
radius with LOS. The 1.05 m standoff is a ranking preference only.

This is a positive control, not a deployable or paper test result. A sealed
benchmark must use a separate scene allow-list and a separate evaluation entry
point after development is complete.
