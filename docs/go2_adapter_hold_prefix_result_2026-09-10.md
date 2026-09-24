# Adapter hold diagnosis: first 1,000 observed decisions

The fixed snapshot completed with result SHA-256
`1a9b9627a9496b40aaf5e20aacad00172fbc892c827bda3e4d830df5b4843884`
in `go2_adapter_hold_prefix_v1_attempt_001` under the navigation artifact root.
Its canonical decoded-row SHA-256 is
`c6ff5bd00ccc2320c7487ecd3de6891504bcaee7f6b57461412dec8b11bcbc2a`.

Independent verification (tool session 17015, exit 0) checked all 1,932 bound
sources and three output artifacts, reconstructed all compact rows and the full
summary from the saved stream, and compared every decoded row with the original
first 1,000 rows. The active batch and queued frontier source bindings also
remained unchanged. This did not rerun model inference or reconstruct raw sensor
processing; the original full episode audit remains pending.

There were 595 discretionary holds outside mission settling. In all 595,
at least one moving action passed the saved raw phase, surface and eight-step
nominal path gates, but holding had the highest saved utility among those
actions. In 566 of these holds, left turn was the only raw-eligible movement,
while the higher-utility right turn was path-vetoed. None of the 595 selected
holds carried an applied residual feasibility recovery. Raw gate eligibility
does not reconstruct any unused corrected recovery paths and does not certify
physical clearance.

The longest uninterrupted hold run was frames 395–470: 76 observations spanning
7.5 simulated seconds. Its observed position estimate changed by 15.1 mm, and
its intermediate target remained `[0.775, 1.175]`. The complete snapshot
contained 22 forward, 141 arc and 238 turn requests, as well as 599 zero requests
(595 discretionary holds and four rows without a selection). No observed arrival
or terminal was recorded. Observed goal distance changed from 4.687 m to 3.378 m.
Native position, contact, command completion and final episode outcome were not
read or certified by this diagnosis.

The existing ten-command infeasibility wait does not address this case: a valid
selected hold resets that counter. The next distinct development candidate is
an explicitly bounded single-command reorientation after repeated discretionary
holds, retaining all raw geometry vetoes and ranking eligible turns with the
same model utilities. A turn may score below holding; whether it enables useful
new observations or merely worsens stagnation needs a raw prefix replay and a
fresh physical experiment. The already queued reached-frontier intervention is
separate and remains unchanged. Independent comparison layouts remain unused.
