# Multi-reference tracking and persistent return intent: development progress

Update: the fresh batch and full raw audit are now complete. See the
[audited result](go2_intent_room_return_result_2026-09-06.md) and
[current next steps](go2_intent_return_to_jepa_next_steps_2026-09-06.md).
It remains0/3 full returns: the left final-home hold passes, but an intermediate
native hold fails; right loses tracking and low friction exceeds the proposed
endpoint excursion limit. The earlier running-status paragraphs below are
chronological records, not current live-job status.

The ultimate RGB-plus-sensor JEPA novel-maze goal remains unachieved. These
changes repair execution prerequisites; they do not replace the learning,
memory, independent-maze or hardware requirements.

## Completed recorded replay

`.generated/go2_multi_reference_recorded_replay_v1_attempt_001/result.json`
has SHA256 `efd4aa06967d78988e5b9f94a621ced26122c55ca75c0d2a27e02e819d519cb8`.
The replay tracked all4986 frames with no missing pose. The nominal-left stream
used one qualified older reference at frame1677; lower-friction-left used one
at366; nominal-right needed none. The4954 comparable valid original prefix
poses matched exactly. Maximum native position errors were8.55 mm left,
14.68 mm right and2.57 mm low friction. Reference decisions used sensor quality,
not native error. These are reused development recordings, not independent
generalization or a physically executed recovery. Previous return success
remains0/3 until a fresh collection and independent audit establish otherwise.

Persistent return intent now separates travel heading from final-home heading
and preserves corner approach heading across clipped legs. Sixteen focused
tests passed. The previous full196-file regression completed2492 tests in
199.17 s. Eleven new external-artifact-root tests also passed separately.

## Prospective successor

The [fresh three-trial protocol](go2_intent_room_return_v1_2026-09-06.md)
combines these fixes with the unchanged empirical coupled-feedback baseline.
It keeps both nominal signed routes and low friction, unchanged goal tolerances,
all failures and bounded command budgets. The external owned artifact root
holds only new data; no previous evidence is deleted or relocated. It retains
a40 GiB recovery-filesystem reserve and uses separate external-data bindings.

Preflight review found that the old cached reader's3606-frame limit was smaller
than the complete initial+3600-command+10-drain population. The successor admits
3611 frames without changing the mission budget. Raw sensor reconstruction now
uses copied cached packets, retaining the original native checks. Thirty-nine
focused tests cover the new source and predecessor observer/intent/root guards.
Read-only preflight verified671 sources and59658 input identities. The latest
198-file regression completed2515 tests in198.21 s. The physical batch was
launched under handle41338 with launch SHA256
`7a5c427ca521de367a301376fafd262aeda5f6e16b7ed876f2b40e87ce0b1a91`.
It remains an ongoing collection, not an audited result. Latest live progress
and the exact handle are recorded in the autonomous checkpoint.

The first fresh nominal-left trial reached a seven-stage ROOM_RETURN_CANDIDATE
at tick2412. Final visual position error was0.0534837 m and heading error
0.0431962 rad. These are controller estimates, not independent native acceptance.
Right and low-friction trials and the full batch audit remain outstanding.

## Exact JEPA target pairing

The distinct [recorded pairing diagnostic](go2_pulse_timed_pairing_diagnostic_v1_2026-09-06.md)
completed all185 old pulse windows (61 left,109 right,15 low friction), retaining
all failures. All have actual four-frame history. Of925 prospective target
slots,917 have actual RGB metadata and fully executed matching command prefixes;
8 are censored as unexecuted prefixes. Another555 slots are explicitly unknown
future-plan padding. Exact minimum-braking endpoint coverage is183/185 pulses.
There is no promotion of unavailable targets to observations or substitution
of2.5 s for the2.2 s short-pulse endpoint. Eleven focused tests passed, including
materialization of actual recorded RGB/body tensors and wrong-timestamp rejection.

The result SHA256 is
`e8786cbda4b821d79bf8554311423224562e04ddf0bcf78fff960fbd9c581e84`;
the185-window index SHA256 is
`7798f2f8b60491475828d900c9df75656bf5ad08eacc0d51850615b53c61b900`.
This completes the observation/prefix portion of the adapter, not target-only
native motion/contact reconstruction, matched partial-timing losses, independent
layout data, training or a causal JEPA result. The current model is still untrained.

## Required next scientific work

1. Execute the bounded fresh protocol, then independently reconstruct sensors,
   command tape and memory and score all native local/winding/home holds. Do
   not infer success from stage counters or visually estimated home alone.
2. If incomplete, localize failures in sensing, goal semantics, action coverage,
   model transfer or bounded search. Preserve source/result identities and use
   distinct successors. Do not loosen6 cm/.05 rad criteria to erase failures.
3. Once continuous execution works, use actual branch/marker observations and
   persistent visit/attempt memory to explore and physically backtrack in
   connected mazes. Scripted room waypoints do not demonstrate useful memory.
4. Finish the real-data pulse-timed JEPA adapter: exact target timestamps,
   verified executed known-command prefixes, separate raw-observation and
   physical-outcome censoring, and target-only native labels. The current
   pulse-timed model remains untrained and outside the controller.
5. Collect adequate action/state/scene coverage and freeze independent layout
   splits before fitting. Compare empirical, direct supervised, supervised
   rollout and JEPA with matched data/sensors/actions/budgets and multiple
   training seeds. Separate predictive-training, online-rollout and memory
   effects; test actual mission completion rather than latent loss alone.
6. Address hidden-robot ideal imagery, sensor calibration/noise/dropouts,
   observed swept-body clearance and actual compute deadlines; obtain bounded
   hardware evidence when available. Current paused simulations are neither
   realtime nor deployment-qualified.
