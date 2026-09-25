# Navigation testbed and controller capability: frozen preregistration

Authority: [the approved brief](go2_navigation_testbed_controller_capability_agent_brief_2026-09-25.md) and Andrew Knowles's explicit kick-off on 25 September 2026. This replaces the September 7 goal and the closed decision-headroom programme. No new simulation, fitting or set generation preceded this freeze.

The executable specification is [the JSON record](go2_navigation_capability_preregistration_v1_2026-09-25.json), SHA-256 `b6a84db2f304282d9d7ec9327ac21420d6e8c3e262de20178c7f243791a887d6`. It binds the deployed V4.2 source stack and models. All recorded source bindings match the working files at freeze.

| Item | Fixed choice |
|---|---|
| Geometry | Existing 4×4 tree-maze generator; 1.3-m pitch, 80-mm walls, 1.4-m height; construction seed 2026092501 |
| Sets | First 10 accepted layouts development, next 20 validation, last 60 sealed; exclude prior registry and all eight audit layouts |
| Episodes | Two predetermined seeds per maze; wall-occluded home/beacon pair, endpoints at least 0.50 m from walls, inflated geodesic strictly greater than 5.2 m; seeded initial heading |
| Mission | 480 simulated seconds plus 1.5-s settling; existing coordinate instruction in the initial body frame; existing verified arrival reader |
| Arrival | Observed radius 20 mm; physical radius 40 mm throughout one-second dwell; 100-ms speed no greater than 0.05 m/s; zero requested command during dwell |
| SPL | True geometry, 0.46-m disk inflation plus 5-mm clearance, 20-mm grid; native 2-ms actual path length; failure gives zero |
| Stall | Final selected hold / planning decisions with a selection, by mission phase; overrides separately reported |
| C3 | Maze-data head, fixed from existing lower regret point estimates in both audit exposure strata; this does not claim superiority |
| C4 | One direct supervised fit: same readout training populations and targets, frozen V-JEPA preprocessing, causal context and candidate command tape, approximately 17.4M trainable parameters, one seed, 1,760 updates, final checkpoint |
| Harness | V0 deployed V4.2 stack; at most six versions including V0; one shared change per version |
| Gate | C1 9/10 triggers C0 gate; C0 at least 19/20, zero contacts and zero hard-clearance violations |
| Capability | At least 80% round-trip success with zero disallowed contacts; 40 validation episodes/controller, C0 lowest 10; 95% maze-cluster bootstrap intervals |
| Caps | 120 wall-hours runs/videos; C4 12 GPU-hours including encoding; 2-GiB VRAM reserve per device; RecoveryStorage 12-GiB and workspace 4-GiB reserves |

C4 inventory found an older supervised rollout interface and a readout requiring future features, neither an unchanged direct substitute for the current prediction slot. The fixed new fit uses the exact 5,966 original/heading and 2,448 maze contexts of the chosen head. Missing causal training inputs are reported, never repaired with new evaluation observations. Historical training-render provenance remains unverified for C3 and C4.

The mission uses the deployed positional instruction, with the beacon hidden by walls at the start. It introduces no visual beacon detector, true map, true current pose or new sensor into the controller. Test episode packets are reserved because kick-off requests episode registration for every set; they grant no E1 execution authority.

The first development episode supplies the C0 executed-prefix fidelity check and serial throughput pilot. The video pipeline is checked on a labelled pilot replay. Concurrency and caching require identical decisions and trajectories before use. The budget projection first reduces validation to one episode per maze if necessary; if that still exceeds 120 hours, work stops with the measured blocker.

All failures are retained. Validation is used only after the shared harness passes its gate. Official video selection is fixed in the JSON before viewing: lowest common successful episode, otherwise each controller's lowest success, otherwise its lowest episode labelled failure. Deliver the capability report, videos and E1 proposal, then stop.
