# Navigation capability kick-off: development progress

The authoritative programme is the [25 September brief](go2_navigation_testbed_controller_capability_agent_brief_2026-09-25.md). This is a progress record, not the final capability report.

Pre-registration was committed as `bb9ab232`, before generation or physics. The 10 development, 20 validation and 60 sealed test mazes and fixed episode packets passed structural registration; initial harness/source registration was committed as `523005c1`. No validation or test episode has run or been rendered.

## First oracle development pilot

| Quantity | C0, development maze 00 / episode 0 |
|---|---:|
| Native beacon arrival | Passed, 64.8 s |
| Native home arrival | Passed, 114.8 s cumulative |
| Recorded mission duration | 114.82 simulated seconds |
| Episode wall time | 676.10 s |
| Wall seconds / simulated second | 5.89 |
| Outbound / return SPL | 0.836 / 0.914 |
| Disallowed contact samples | 0 |
| Hard / operating-margin violations | 0 / 0 |
| Unresolved sampled clearance | 0 |
| Minimum all-primitive clearance lower bound | 0.17273 m |
| Matching candidate-prefix comparisons | 1,668 passed |
| Maximum matching-prefix position error | 0 m |
| Median / p95 planning latency | 1.677 / 1.735 s |
| Hold plans, outbound / return | 2/156 / 1/122 |

Both the existing initial-frame physical arrival criteria and distance to the generated fixed world beacon/home pass. Articulated clearance was evaluated at every native 2-ms step, with all 27 collision primitives. The FK interval robustness check also has no threshold failures. These results establish one successful development episode; they do not pass the 19/20 oracle gate or establish unseen-maze capability.

The oracle uses only true candidate motion in the prediction slot. The unchanged controller receives its ordinary sensor packets, measured pose, observed map and mission instruction. Matching-prefix checks compare every native sample under the same applied tape; an override or subsequent command change terminates that matching prefix.

Runtime evidence is under:

`/home/andrewknowles/RecoveryStorage/LeWMQuad-v3/.generated/navigation_development_artifacts_v1/go2_navigation_capability_v1_attempt_001/runs/v0_pilot_C0_dev00_ep0_attempt002`

See `episode_evaluation.json`, `oracle_prefix_check.json`, `native/physics_trace.npz`, `native_clearance_summary_arrays.npz`, `planning.json`, `requests.json` and `result.json`.

## First command-history development pilot

C1 also completed development episode 00/0 on the unchanged v0 stack. Beacon arrival was at 68.5 simulated seconds and home arrival at 111.2 seconds. The complete recording took 905.59 wall seconds (8.14 wall seconds per simulated second). Outbound/return SPL were 0.821/0.928. There were zero disallowed contacts, zero hard or operating-margin violations, and no unresolved sampled-clearance intervals; the minimum articulated separation lower bound was 0.15585 m. Hold decisions were 2/166 outbound and 2/104 on return. Median/p95 planning latency was 2.607/2.668 wall seconds.

The source RGB and depth hashes were retained successfully for this episode. The original recording occupies approximately 176 MiB. Its replay pipeline test also tests prospective omission of the neural workload that the deployed C1 selector does not use; that optimisation requires identical command-model predictions, decisions, commands and native physics records before admission. The full original neural workload remains the measured baseline above.

Evidence is in `runs/v0_pilot_C1_dev00_ep0_attempt001` under the same fresh output root. This is a development pilot, not a capability estimate or a completed ten-episode harness screen.

## Preserved implementation issues

The first oracle attempt stopped after settling, before any mission command or branch, because its new `nn.Module` retained the default training flag. The evaluation-mode correction and fresh attempt were recorded in [adapter revision 1](go2_navigation_capability_harness_v0_adapter_r1_2026-09-25.json). This changed no controller algorithm.

The successful pilot's owner called physical persistence but omitted the separate observation-persistence method. Physics, decisions, commands, observed poses and the initial snapshot survive; the original camera images and their per-frame hashes do not. Its physical arrival results use an exact timestamp join between retained acquisitions and native samples, with the unchanged dwell/speed/zero-command thresholds. This reconstructs no image evidence. This pilot cannot provide a source-image-verified video.

[Adapter revision 2](go2_navigation_capability_harness_v0_adapter_r2_2026-09-25.json) invokes the unchanged observation writer. The original pilot is preserved and is not repeated. The pipeline video is assigned to the first C1 pilot before that run begins.

C4's first startup stopped before encoder calls or optimiser updates: `torch.__version__` is a string subclass rejected by the strict converter comparison. The successor records it explicitly as a native string; the converter and fixed training plan are unchanged. The failed root is preserved and its elapsed GPU-owner accounting carries forward.

## C4 and remaining work

All 8,414 prescribed training contexts have their causal RGB and command histories: 5,966 original/heading contexts and 2,448 maze contexts, from 156 training-role recordings. There are 9,948 unique causal image paths. No new development, validation or sealed material enters fitting. The direct model has 17,397,283 trainable parameters; a synthetic CPU check verified its candidate/horizon interface. Training-render provenance remains unverified.

The fixed fit is in progress under its 12-GPU-hour cap, with no simultaneous navigation GPU owner. Next come the remaining serial pilots, verified C1 pipeline video, concurrency/equivalence checks and measured budget projection. Harness iteration, the full oracle gate, validation capability results, official videos and the E1 proposal remain outstanding.
