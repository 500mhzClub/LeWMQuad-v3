# Live RGB-history preparation and recorded-stream replay

The new `OnlineRGBHistory` supplies four chronological actual packets to the
temporal model interface. It has no model, planner, world pose, maze graph or
persistent place memory. It must receive every100-ms packet, including between
action decisions. It does not itself issue physical commands.

Nine focused tests pass: live/offline tensor equality, continued sliding updates,
gap re-warmup, stale/duplicate/foreign-episode rejection, invalid or rewritten
sensor samples, privileged-field rejection and isolation from caller mutation.
Invalid input clears readiness. A gap requires four new consecutive packets;
stale tensors cannot be retrieved as a new decision. A future controller must
catch input faults and explicitly request safe stopping—it must not silently
continue its last motion command because this buffer raised an exception.

## Actual-data replay

Replay of all120 audited development policy streams passed:

- 8,943 command-clock packets pushed;8,583 four-frame-ready packets.
- 818 derived windows match the offline temporal tensor interface exactly.
- 96 noncanonical initial sibling contexts deliberately excluded from that
  equality comparison. Their actual initial images are not replaced by the
  canonical training stop-branch image.
- 23 off-command-clock terminal frames retained in the source corpus but not
  pushed as ordinary command-boundary inputs.

These counts account for all8,966 corpus images and all914 derived windows
(818 compared plus96 explicitly excluded). The comparison preserves the known
distinction between canonical training contexts and actual online RGB bytes.
There were no new simulations, learned policy choices, optimizer updates,
checkpoints or hardware commands. The replay validates data-path equivalence,
not safety, perception quality or repeated navigation performance. Terminal
contact recognition is not an input-buffer function.

The contract remains complete, current, zero-latency ideal simulated sensors.
Real delay, dropout, calibration and clock synchronization need separate work.
This component is preparation for a subsequent fixed sensor-only replanning
study, after the running temporal comparison completes and passes audit.

## Identity

Root: `.generated/go2_online_rgb_history_replay_development_v1_attempt_001`.
Session20826 terminated exit0, PASS120 streams. Test session49198 exit0,9 passes.

- Launch SHA-256: `c5cfb97ee0d60acdfbede50ea41b00b05833fbf99ca1d077324560c679a6c890`.
- Result SHA-256: `587653dd19100711269df088c4bfd1ac0c3b6917844fb0038a0c86974b090edc`.

The replay binds its three new source/test/checker files and consumes the audited
policy-only loader; this is not a recursive source-export certification or
deployment approval. Preserve this evidence and separately bind dependencies
when integrating the next actual controller. Final maze/JEPA/hardware claims
remain open.
