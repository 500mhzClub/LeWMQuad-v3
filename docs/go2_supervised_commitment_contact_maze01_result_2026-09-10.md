# Supervised commitment-horizon contact maze1 result

Complete pilot90116b248154b9d4731501d5c6cc6a317345101b9f39c6ed5ce712c9baa32986,
root go2_supervised_commitment_contact_maze01_pilot_v1_attempt_001.
Status SUPERVISED_COMMITMENT_CONTACT_MAZE01_PILOT_V1_COMPLETE.1822 sources,
945 outputs; pilot wall852.3424074030481s. This is completed raw-audited episode30,
with zero verified round trips. The model is the previously assigned supervised
rollout model; the new expanded-data models were not used.

Original waiter14895/PID2571800 completed0 after parent2633510 exited0. Waiter
resultdc57fe4cd5fb8083ec6ac43ae11d85e6d3e048db62d9b03f06361eb621378536,
SUPERVISED_COMMITMENT_CONTACT_NATIVE_WAIT_V1_COMPLETE,1825 sources/five outputs.
Its original native verifier reexecuted and recorded all raw audits and actual
physical prefix passing. Verification covered137978 unique files and
64,229,075,647 bytes freshly hashed both initially and finally,1,027,220 digest
requests and26 isolated verifier functions. The original queue identity remains
1473b2f801991698b41ddd8d97af627e2b1d34621e1cdc6c48886afb03141111.

Exact bindings:
- launch:f61bbd19f01bde603945de9e2feff6c787bd2c62d87942a7a41967e498052096
- collection result:63e2ca4454b10a5a5c93c93ed36c1432d23a1d3f1167feae9de1b7321ff801fd
- raw audit:eeb9d3570fc312d5d76949680b3261c98be7769f0cd82eed3ad439507e0867da
- prefix:aa619feff75419630f4dd6c265296f0fe89f5be22760295882a2c1e40b440d74
- worker terminal:e6089685618c5f1883808bdab3d0d86fe5f2d4a49b6e945b0d5ab36565d3bb73
- parent stdout:bae900c4a8d032efefd7493d905932c0a7e26675cc8a644397bfa919b909b133

Collection:151 observations,150 completed commands,8250 physics samples and10
zero-drain commands. No physical/acquisition stop. First terminal observation140
retains decision/mission frame139: SENSOR_OR_MODEL_FAILURE, same-episode current
visual evidence required. Last observed goal distance1.2962273155376618m,
OUTBOUND; no observed/native arrival, edge crossing or round trip. Do not confuse
this visual-evidence failure with the recent-reference maze1 floor-registration
failure at1545. No more specific visual failure cause is asserted here.

Raw sensor reconstruction, model replay, commands, unchanged model and strict
visibility pass; hard failed frames[]. Selected actions:13 right arcs,10 forward,
20 left arcs,91 left turns,one right turn,two holds. Warmup/drain/terminal zero
requests are separate. Outbound loop-erased cells remain[[-1,0]], with no crossing;
native terminal quietness fails. These are valid negative execution results,
not a learned-planning, independent-layout or deployment success.

Physical/public prefix:four observations,900 physics samples,one raw model bank
exact; first changed command at3 is the completed right arc[0.16,0,-0.45]
instead of right turn[0,0,-0.45]. Candidate decisions match the prospective prefix.
Raw physical prefix fingerprint8419be1a3143128fcef2a1cf843d6476063177cc52f83316c42fd9b2bc7789fa.
Worker wall660.5599782350473s, peak RSS2,878,152,704 bytes.

Following the authenticated waiter completion, the already prepared isolated
recent-reference direct-flow maze3 runner was started as25801/PID2636286. Its
launch25c34387e44b035f05a4f372ecf70cd337e875de508ba3a9b037ccf741da0eb7
binds1843 sources and the exact completed contact-waiter SHA above. Native worker
2637549/create_time1789018454.4 is live;2637548 is its resource tracker. Preserve
this single native scene. It uses the prospective prefix16c6917d2e4c2b728bd08a330290e141a93a500f9f28b7a3abd27e9d4f51926a
and the already fixed old full-JEPA model, without the partial-floor-height
intervention. No result or round-trip evidence is yet available from it.
