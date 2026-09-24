# Completed diagnosis of the frame-504 floor conflict

The original floor registration rejection was reproduced exactly from the
preceding admitted floor state and the actual paired public packet at frame 504.
The original failure latch, anchor, reference and preceding frame were preserved.
The failure came from auxiliary candidate disagreement with the transported
reference; it was not a missing visual pose or a new direct-flow association at
this frame. No policy or subsequent physical outcome was changed or inferred.

## Identity and scope

- Root: `go2_direct_flow_maze01_floor_conflict_v1_attempt_001` under the external
  development artifact root.
- Result SHA: `d2512d3241e38ac608aa2185cb62e5aafdd4fc13db962fffc68e45d82eaa992e`.
- Launch SHA: `f9cb771321f9e056719cc4716ed5b8b9a3651ca2afed1f944a4c9b581206b70c`.
- Completed native input:
  `d6774bae22cb9effeb0cd85ae255de203de1539541f701d57788b58ab00769de`.
- Session 14291 exited 0; 1,686 source bindings and the output launch binding
  were rechecked after completion. Wall time after admission: 362.720028528 s.

The diagnostic checked ordered original decisions 0–504 and loaded only raw
packet 504. The native input's full raw audit had already reconstructed its
visual evidence, complete controller/model decisions and dispatched commands.
No model or tracker was reexecuted by this diagnostic. All public arrays and
saved evidence remained unchanged; decision/packet 505 was not consumed.

## Actual measurements

| Measure | Primary | Auxiliary |
| --- | ---: | ---: |
| Current candidate points | 0 | 5,956 |
| Points beyond the existing 3 mm gate | 0 | 82 |
| Maximum absolute residual | unavailable | 3.392187 mm |
| RMS residual | unavailable | 1.491898 mm |
| Mean signed residual | unavailable | +1.059830 mm |
| Minimum / maximum signed residual | unavailable | −1.175571 / +3.392187 mm |

The worst auxiliary candidate was sampled at row 246, column 2, body point
`[0.691204146, 0.396495921, -0.272683656]` m. The complete candidate cloud SHA is
`5517e5b7c03ffa8fc9d8842a18fe1718c9f1756e122a5c46e367879c9937ed8c`,
and its sampling-mask SHA is
`ad2872ebafd92c705eaf416e37dfcb0edcac55e862acff2e5a260eb981c130f7`.
All candidates were retained. The absent primary candidates provide no
geometric agreement or disagreement evidence.

The joint fit was unavailable because its second covariance eigenvalue was
0.002387664056787141 m², below the unchanged 0.0025 m² extent gate. Candidate
count exceeded the unchanged minimum of 100. The smallest eigenvalue was
8.287494372272897e-13 m². This coherent but insufficient-extent candidate
population did not qualify as a newly admitted plane under the existing rule.

The retained full floor anchor was frame 490, age 14 frames (1.4 s). Its
transported correction magnitude was 11.554809 mm and angle 0.023868470 rad,
within the existing 5 cm / 0.10 rad development limits. At frame 504, the selected
visual camera was auxiliary with reference 503, and no direct-flow fallback
receipt was present. These facts locate the failing stage; they do not prove
whether the underlying error belongs to visual motion, floor identity or the
transported reference relative to physical truth.

## Bounded next hypothesis

Using this current candidate population only, subtracting its +1.059830 mm mean
signed residual while holding the transported normal fixed would change the
reported residual range to −2.235402 through +2.332357 mm. Thus a scalar height
constraint could, algebraically, make these current points agree within the
unchanged 3 mm gate. This calculation is not an admitted pose, plane, policy
change, uncertainty bound or a physical counterfactual trajectory.

The next candidate should test a separately typed height-only observation when
the full plane lacks two-axis extent: retain the transported normal, require
the original minimum candidate count, use every measured point, retain the
all-point 3 mm gate and total pose-correction limits, and never promote this
partial observation into a fully observed floor anchor. Ordinary full-plane
and successful transport behavior should remain exact. Test coherent height
offsets, tilted/incoherent populations and unavailable measurements, then replay
the complete controller prospectively to its first changed command. Fresh
physical continuation is necessary before any navigation claim.

The goal remains unachieved: 22 completed/raw-audited native episodes and zero
verified round trips. Existing hold and support-cache experiments remain
separate and cannot supply this missing navigation evidence.
