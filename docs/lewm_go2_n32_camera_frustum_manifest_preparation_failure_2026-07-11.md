# Go2 camera-frustum manifest-preparation failure incident

Date: 2026-07-11

Status: `acknowledged_pre_authoritative_run`; recorded before any retry,
authoritative audit, label-shard byte open, or audit result. This document is
the controlling combined pre-authoritative access record.

## Earlier incident incorporated

The earlier out-of-ledger search incident remains recorded verbatim at
`docs/lewm_go2_n32_camera_frustum_observability_preflight_access_incident_2026-07-11.md`,
SHA-256
`683fd43e68f9121f3b4937fbebbf01f760d46ccbc90ff4d1d7551b9a251184ca`.
Its disclosure, restrictions, and withdrawn zero-preflight-access claim remain
in force. This combined record incorporates that exact incident by path and
hash, then adds the failed metadata-preparation access below. Neither incident
is evidence or scientific licensing.

## Failed command

At approximately 2026-07-11 18:13 BST, after the binding and reviewed human
implementation manifest had been frozen, the first authorized metadata-only
preparation command ran with:

```text
PYTHONPATH=/home/andrewknowles/Workspace/LeWMQuad-v3:/home/andrewknowles/Workspace/LeWMQuad-v3/lewm_worlds /usr/bin/python3 scripts/audit_go2_n32_camera_frustum_observability.py --authorization-sha256 6b8a243d8ec2d3fa1df386defb761f2defe87d5ed491e371df050d3054e644eb --prepare-manifest-inventory --human-manifest-sha256 ef8d1a8a768c430caad82505634ec7e25e703c50c4b4a8d098b7a41267b113e6
```

The process exited 1 and emitted no preparation inventory. Shell redirection
created `/tmp/go2_camera_manifest_inventory.json`, but it contains no runner
result and is not an audit artifact.

The exact failure was:

```text
ValueError: fit panel endpoints do not equal the frame selection one-to-one
```

It occurred in `_validate_frame_selection_and_rendered_set`, called from
`_read_source_geometry`, before the preparation pass returned its phase ledger.

## Access accounting

This was metadata-only contact under the then-current authorization. Before
the exception, the runner:

- hashed the frozen implementation/document graph;
- checked the exclusive result path;
- parsed the fit panel;
- inventoried label-shard paths and commitments without opening an NPZ member;
- parsed an initial allowed prefix of committed physical geometry,
  render-audit, render-summary, and frame-selection JSON metadata.

Because the exception occurred before the preparation result was emitted, the
exact in-memory phase ledger is unavailable. No count more precise than this
ordered code-path boundary is claimed.

The failed command did not open a label-shard byte, decompress an array, stat
or hash an RGB image, decode a pixel, inspect a model/checkpoint/output, or
access G2, selection/calibration roles outside the committed V04 fit-selection
metadata, physical-nontrain, runtime, held-out, seed-20260711, or sealed
payloads. It wrote no repository file and no immutable audit result.

## Consequence

The prior pre-run authorization and human implementation manifest are stale.
No real metadata retry is allowed under their hashes. This incident is not
scientific evidence and cannot tune a label or model gate.

The failure exposed a protocol error: the V04 selection/render artifacts
describe the larger committed rendered corpus for each source scene, while the
fit panel selects 320 unique endpoints from that corpus. The correct
provenance relation is:

1. the frame-selection keys and rendered-summary keys reconcile exactly over
   their full committed per-scene set;
2. the selection artifact binds the global source-row file, while the original
   render plan and summary bind the larger source frames JSONL and its camera
   contract; neither larger file is required to equal the selected key set;
3. every selected render key, and every fit-panel endpoint as a subset of those
   keys, occurs exactly once in the source JSONL with the same timestamp, while
   every fit endpoint also matches the rendered image SHA-256;
4. no fit endpoint is missing or repeated, and unrelated committed source
   frames are parsed and ledgered but never used as fit-label evidence.

A dated binding amendment must freeze that relation, bind this incident, and
restart the preparation ledger from zero before any retry.
