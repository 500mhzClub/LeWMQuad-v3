# Go2 N32 camera-frustum audit preflight access incident

Date: 2026-07-11

Status: recorded before any authoritative camera-frustum audit run. This
record supersedes every earlier claim that no fit-panel or prior-result bytes
had been read during camera-audit implementation.

## Incident

During source-provenance investigation, the root agent ran:

```text
rg -n 'render_audit|source_index|frame_selection' .generated/go2_physical_micro_overfit -g '*.json' --files-with-matches 2>/dev/null | head -20
```

The command ran shortly before the incident was recorded at
`2026-07-11 14:15:31 BST`. The execution timestamp was not independently
logged, so no more precise time is claimed.

The glob contained exactly these two JSON files:

1. `.generated/go2_physical_micro_overfit/patch7_v1/panel.json`, frozen file
   SHA-256
   `c3f44c6b1147efbb6a5fbc2294c6431c72e25da877cab6884972d25c1ffdb16c`;
2. `.generated/go2_physical_micro_overfit/patch7_v1/seed_20260710_result.json`,
   frozen file SHA-256
   `6e2aacd18fe1d692fb6ad682b41132563dcbcdb95c7b7ce719f407baf6c91a8c`.

`rg --files-with-matches` opened each in-scope file and scanned an unknown
prefix or all of its bytes as needed outside the audit runner's access ledger.
Standard output contained only the already-frozen panel path. Consequently
the only positive new observation was:

- at least one of the three literal search tokens occurs somewhere in the
  panel bytes.

No prior-result path was emitted. Because standard error was suppressed, no
content-absence inference is claimed for that file.

No JSON object, row, numeric value, class label, metric, prediction, or model
parameter was printed or parsed by the agent. The command did not open a label
shard, source-geometry file, RGB/image file, holdout, physical-nontrain role,
G2 artifact, runtime artifact, sealed artifact, or seed-20260711 artifact.

## Classification

The panel scan was an allowed-role input read performed too early and outside
the frozen ledger. The prior-result scan was a forbidden model-result byte
read under the superseded preflight boundary, even though no result value was
exposed. Both must remain disclosed; neither may be folded into the fresh
authoritative runner ledger or described as zero access.

The incident did not reveal the answer to any audit gate. Geometry, support,
provenance, reconstruction, and authorization requirements were already
ordered by the frozen design and adversarial review. It therefore does not
license a scientific result, but it also does not require opening or replacing
any fit label, holdout, G2, or sealed payload.

## Required handling

- Freeze a superseding audit binding that names and hashes this incident
  record before any parsed panel or label access.
- Start the authoritative runner and finalizer ledgers from zero after that
  superseding freeze, while reporting this preflight incident separately.
- Make the machine-readable implementation manifest and torch-free finalizer
  bind this record and require status `acknowledged_pre_authoritative_run`.
- Never restore the withdrawn zero-preflight-access claim.
