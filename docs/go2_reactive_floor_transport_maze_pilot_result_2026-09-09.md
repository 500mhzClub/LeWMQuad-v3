# Reactive floor-transport maze0 pilot result

The reactive method completed and passed raw auditing, but reached neither
the outbound goal nor a verified round trip. It stopped at observation273
with `NO_REACTIVE_ACTION_SATISFYING_CURRENT_GEOMETRY`, no exception string,
and a zero request. Observed goal distance at the stop was2.8197429239327283m.
This is a valid negative development result on reused maze0.

Root: `/home/andrewknowles/RecoveryStorage/LeWMQuad-v3/.generated/navigation_development_artifacts_v1/go2_reactive_floor_transport_maze_pilot_v1_attempt_001`.
Session18370 closed normally with exit0. Final result SHA-256:
`4d377b9ec202c96099615ca5e3679037d5cb81d30c319afdd7febb9b9f0c9837`.
Launch SHA-256:
`f3408677bddb3fdfe4e82be4503b1f67dc23d039bb0524a52b813a51b4c1fce3`.
The final result binds1675sources and1743artifacts. Runner wall time after
launch911.3410807489417s, worker733.3451112201437s, peakRSS3,289,071,616bytes.
The runner verified all inputs, sources and artifacts before completion.

Collection contains284paired observations/decisions,283completed commands,
14900physics samples and10zero drain ticks. Physical/acquisition stops are
absent. Raw sensor reconstruction, controller-command replay and command
audit pass; strict physical visibility passes with no hard measurement failures.
Native evaluation records two crossings on declared-open edges:
`[-1,0] → [0,0] → [0,-1]`, with no invalid crossing. Arrival windows are empty;
return traversal, physical retrace and native round-trip success are absent.

The prospective intervention was physically reproduced:4paired prefix frames,
900physics samples, first intervention at frame3, unchanged prior requests,
exact shared observed state and complete candidate decisions. The intervention
command was completed. Raw physical prefix SHA-256:
`8419be1a3143128fcef2a1cf843d6476063177cc52f83316c42fd9b2bc7789fa`.
Post-intervention physical outcomes are this fresh execution's evidence;
they were not inferred from the learned predecessor tape.

Artifact SHA-256:

- Collection `reactive_floor_transport_novel_maze_00/result.json`:
  `560870fbc325d00d113a95c2c62879148012f0c49489a241e01b6434e93accd9`.
- Raw audit `reactive_floor_transport_novel_maze_00_audit.json`:
  `d458a25a710229d4ab12e9736e0dc5e5eb8b9f00859cbedc2c695cf2adb23b92`.
- Worker terminal `reactive_floor_transport_novel_maze_00_worker_terminal.json`:
  `165391c926ba90540ee68b8f7da8a95eae0b10b116905cf1e0fed153ac4701d3`.
- Decision stream `reactive_floor_transport_novel_maze_00/context_decisions.jsonl.gz`:
  `1d060370f34a9e2990b3d1e112a1b73a129204f68938fdbed5d022b1ee815fb3`,
  rehashed unchanged around bounded terminal inspection.

This comparator uses the shared observer, registration, map and settling
mission with a current-geometry action rule, no high-level learned model,
candidate future predictions or learned residual. It is a method comparison,
not an isolated prediction-ranking ablation, because predictive feasibility
gates differ too. This single reused layout does not establish JEPA, planning
or memory advantage, statistical reliability, real-time or hardware validity.

Next processing was submitted using the exact final identity: paired maze0
readout session59988, with learned readout
`a46e6051b125347804df4aab948e68a6466007952f7529b4f316ae39b114728c`;
and fixed independent reactive cohort session18182, with completed learned
cohort `a0ac72330a6c34fbae7812a360b22189086f16bd83f586d71e769539ddc2e720`.
The readout does not feed cohort selection or configuration, so its CPU work
may overlap the next single native scene. The existing experiment definitions
and fixed case order remain unchanged.

Hardware refreshed before submission:16physical/32logical CPUs, all32affinity,
3.4%CPU busy,81.019GBavailableRAM,90.872GBartifactfree,21.360GBworkspacefree,
bothGPUs0%busy. The only substantive competitor then was the sequential
feasibility replay at2.561GBRSS; the prior native parent/worker had exited.
The cohort independently refreshes32GiB memory and remaining-case disk
admission, retaining40GiB reserve plus11GiB per remaining case. One native
scene and one CPU thread per job. These are capacity checks, not OS quotas.
