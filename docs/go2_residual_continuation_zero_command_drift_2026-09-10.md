# Residual-continuation nominal dead end during zero-command drift

This is a posthoc diagnostic of the completed reused-development maze2 pilot
818a598ca6336866cf5f4768c11edaf67c8c1ca60896c305f93f69fd0ed5230c.
The native sensor/model/command and strict visibility audits passed, with no
arrival or round trip. Native pose is used below only to evaluate the recorded
motion; no controller, coefficient, threshold or completed outcome was changed.

The last nonzero request before the terminal waiting sequence is left turn at
243. At244 the selected hold is nominally feasible: current clearance is
0.4532428276950597m and the raw hold forecast's minimum path clearance is
0.45246521289813973m against the unchanged0.45m radius. At245 clearance is
0.4513425534578631m but every predicted candidate path fails the radius check.
The first-interval residual fallback also returns no eligible action. At246,
after another zero request, measured-map clearance has fallen to
0.4496244227539443m and the planner enters VIEW_ACQUISITION. It remains
infeasible through terminal255, where clearance is0.44410933877138264m.
The nearest observed cell remains[13,11] in these receipts; unchanged cell
counts do not prove an unchanged complete occupied-cell set.

The existing ViewReentrySelector already tries translating nominal recovery
while acquiring a view. Its rule requires every one of the eight predicted
segments to preserve current clearance and the first endpoint to improve it.
At246..255 every raw candidate path predicts a clearance reduction. The
recovery rule therefore cannot admit one. This is stronger evidence than
merely observing the original scan phase's hold/left-turn/right-turn allowance.
It does not prove that all physically possible recovery actions are unsafe.
The fixed six-action bank contains no reverse or lateral command.

All requests244..254 were[0,0,0], with actual command completion verified.
Over observations244→255 (1.1s), initial-body-frame displacement was:

| Quantity | X, m | Y, m |
| --- | ---: | ---: |
| Observed registered pose displacement | -0.010155268169736997 | 0.005526808958901619 |
| Recorded native displacement | -0.010124555410206064 | 0.0054410607447133936 |
| Observed minus native displacement | -0.00003071275953093339 | 0.00008574821418822539 |

The native drift magnitude is approximately11.5mm; displacement disagreement
is approximately0.09mm. Absolute observed XY pose error remains about3.2mm:
3.256232579107254mm at244 and3.168169838188399mm at255. Thus the displacement
comparison supports actual zero-command drift, not a new large tracking jump.
It is not a validated global pose bound or physical-clearance certificate.

Read-only check93998 exited0 after checking the exact physics, closed decision
stream and command tape before/after, comparing raw sample749+50i endpoints,
using the existing rotation_xyzw function and asserting every zero command's
completion. Relevant bindings under the completed case directory:
- physics_trace.npz:6b046be8d2c8fa2bbc1513b6355a741c58db700cf9547c5138013547f1e0ba57
- context_decisions.jsonl.gz:0e81124252ccdda87c252a8b2ac55613c6bf7ab505110af12fa1e30903add118
- command_tape.json:aea133d89ad7a18e930245871e326c2f274883e4c1a2a14583aa4b7d9c96a4c7

Next work should improve prediction of stopping/transient motion and examine
recovery action coverage. Do not replace this failure by relaxing clearance,
fitting on navigation outcomes or claiming a tracking fix solves it. The
training-only all-phase census and target derivation are recorded separately.
