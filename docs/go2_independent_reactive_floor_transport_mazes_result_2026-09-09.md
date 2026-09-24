# Independent reactive layouts1–3: complete, no arrivals

Session18182 exited0 after all three fixed cases and final verification.
Result `51fa637e8a70bbefc088067c548eb0e50149b490b02a97af04cb2fba720418bd`;
launch `159e55ad6e6373b9a951cb5eb83830bcfce39cb5a5372ccd24c06ca29c29587a`.
Root `go2_independent_reactive_floor_transport_mazes_v1_attempt_001` under the
navigation development artifact volume.1,682frozen sources,4,292artifact
bindings,3307.7784918081015s after launch. No case was selected or replaced
based on its outcome; each used a fresh reactive controller/memory process.

| Layout | Paired observations | Completed commands | Physics samples | First terminal observation | Outcome |
| --- | ---: | ---: | ---: | ---: | --- |
| 1 | 171 | 170 | 9250 | 160 | Visual tracking failure; decision tick159 |
| 2 | 408 | 407 | 21100 | 397 | No action satisfies current geometry |
| 3 | 117 | 116 | 6550 | 106 | No action satisfies current geometry |

All three have ten terminal zero ticks, no physical/acquisition stop,
raw-sensor/controller/command audit PASS and strict physical visibility PASS.
All three have zero native arrival windows, no physical return/retrace and no
verified round trip. Parent2475541 and final worker2479656 exited normally;
do not poll or restart18182.

The paired learned cohort is bound by result
`a0ac72330a6c34fbae7812a360b22189086f16bd83f586d71e769539ddc2e720`.
It likewise has zero arrivals and round trips on layouts1–3, with all raw and
strict visibility audits passing. On layouts1and3 both methods' native paths
remain in the start cell. On layout2, reactive visits[-1,0],[0,0],[0,1],[-1,1],
[-1,2], crossing four declared-open edges; learned visits[-1,0],[0,0], crossing
one. Neither has an invalid crossing or an arrival. These evaluator-only
observations do not enter either controller.

This completes the fixed independent method comparison but does not establish
a learned-planning advantage. The methods share the admitted scene, sensor,
actuator, mission and observer definitions, while their decision policies and
predictive feasibility checks differ. It is not an isolated prediction-ranking
ablation. Both retain observed memory; this is not a memory comparison or a
JEPA objective comparison. Different trajectories and durations also prevent
interpreting whole-cycle timing differences as isolated model inference cost.
There is no reliability, real-time, hardware or deployment qualification.

| Layout | Collection SHA-256 | Raw audit SHA-256 |
| --- | --- | --- |
| 1 | `052fdd1233deb9539dfb6fe15a0fccb02ee436f02d722e33b5f930f822f63cd8` | `b82434126766eb5904f3d4851b1e857ad8e9a2aabe7039780d909c75f750feaa` |
| 2 | `2dfd7c72619a8bd5fd492b42a2888341b674844a3df74f5a79eda334b707722b` | `e8910c8a5e1d783b8a7dd0f09d8454b7976fe76eb18874ffff72a5e3f3a2af5f` |
| 3 | `983bcc1f745af0157576d7344509973e70911ef2ce40660dbad442d902bab248` | `ff91d55a823c286b5fb9f99b374b7b0469e7467ca8cfea426c19f67afdb645e2` |

Worker terminal hashes1/2/3:
`c5fd185ae6329ae6f54e98980f785369415fc8539e1dcabf371d94e3ea829a76`,
`2d7a8a5dd4f099cef69c9d98ecfd787632f84974e72baeaf36f6bf04debcea06`,
`d3e8d92c3075d9e2ae16ce8d33f4e2262917de4aa02de26455714b3567ac8927`.
Final progress hash
`22655f11341ca653712ab12e8db25744f4ec67d5d7ba1e274bd478739edebbc5`.
Final worker wall836.5872598551214s,peakRSS2,824,138,752bytes.

There are now19completed/raw-audited native episodes overall and zero verified
round trips. The prepared planning-memory maze0 pilot follows this completed
cohort, then the residual-feasibility maze2 pilot. Tracking-prefix V2 has now
completed separately and requires a new native collector/audit/launcher before
physical validation. Preserve all failures and scientific negative results.
