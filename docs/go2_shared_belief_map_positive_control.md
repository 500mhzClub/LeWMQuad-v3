# Go2 shared belief-map positive control

`lewm.planning.exact_occupancy_belief_adapter` is a privileged development
adapter for testing the shared `OnlineBeliefMap` contract. It loads exact
manifest occupancy at the canonical 0.10 m online resolution and does not
import the closed-loop benchmark monolith, renderer, perception stack, or any
learned model.

The adapter:

- uses the geometry contract's 0.20 m configuration-space inflation;
- aligns the belief-map origin with the scene occupancy raster;
- loads every selected cell as confirmed free or confirmed occupied;
- configures eight-connected planning without diagonal corner cutting;
- routes only through `OnlineBeliefMap.shortest_path`;
- obtains exploration boundaries only from `OnlineBeliefMap.frontier_cells`;
- requires each selected claim endpoint to satisfy true distance, point-geometry
  line of sight, and connectivity in the independent 0.05 m oracle grid.

Run the public v3 development panel:

```bash
python3 -m lewm.benchmarks.go2_belief_map_positive_control
```

The command reads only `validation_scenes` from
`config/go2_generalization_v3/development.json`. It verifies the geometry hash,
semantic scene-manifest hashes, families, and beacon counts, and refuses paths
labelled as sealed or final evaluation data.

Current development artifact:

- path:
  `.generated/oracle_positive_control/generalization_v3_development/shared_map_report.json`;
- SHA-256:
  `cc34c25bced1b62bc6e889f534232ae9d6e519fc302075cbd559a1ded5d70f8c`;
- 24/24 scenes have exact online-component agreement;
- 24/24 partial-reveal probes have frontier agreement, covering 201 nontrivial
  frontier cells with zero mismatches;
- 96/96 beacons retain a true, oracle-connected claim endpoint and a
  confirmed-free shared-map route;
- no online map cell is absent from the projected 0.05 m oracle component.

The 0.10 m and projected 0.05 m components are not cell-for-cell equal. Across
24 scenes, 7,892 projected fine-grid cells are omitted by the coarser raster;
mean component Jaccard is 0.907781 and the minimum is 0.798826. The mismatch is
strictly conservative in this panel: there are zero online-only cells, so it
removes narrow boundary space rather than introducing shortcuts. Training and
evaluation should therefore use the online raster for map-conditioned labels
and report this resolution loss instead of claiming exact area equivalence.

This positive control validates the shared-map seam only. It does not alter or
promote the learned closed-loop benchmark.
