# Corrected JEPA floor-coverage diagnostic result

The complete 108-observation JEPA run replayed exactly, with the original model
and correction state unchanged. At terminal tick 97, all six candidate
front-right-foot intersections remained without complete retained-floor
coverage at every checked subdivision. This negative diagnosis supplies no
basis for adopting a union-coverage exception or removing the surface veto.

Each row covers the same full 44-mm enclosing nominal foot square. Every tile
uses the original measured-pixel checks and may use any causally retained
observation. Tile boundaries overlap conservatively through outward rounding;
no missing tile is filled or omitted.

| Candidate | Covered of 1 | Covered of 4 | Covered of 16 | Covered of 64 |
|---|---:|---:|---:|---:|
| Hold | 0 | 1 | 6 | 35 |
| Forward | 0 | 2 | 11 | 51 |
| Left arc | 0 | 3 | 13 | 54 |
| Right arc | 0 | 1 | 8 | 40 |
| Left turn | 0 | 1 | 10 | 44 |
| Right turn | 0 | 0 | 4 | 23 |

The 98 causally available frames reproduce the original whole-foot witnesses
exactly. Positive finest-grid tiles obtain witnesses from frames 18–22; none
of the six squares is completely covered. The test is a sufficient nominal
coverage condition. Failure does not prove absence of every possible coverage
proof, identify physical terrain under unknown pixels, or certify collision.
It does rule out this finite, predeclared subdivision method as a demonstrated
remedy for the recorded terminal state. No controller or recorded outcome was
changed, and no prospective native scene was run.

The preceding JEPA controller reached this coverage boundary after 97 ticks;
the direct controller separately failed nominal clearance at tick 42. Further
translation-bias fitting cannot be inferred as the remedy for either failure.
The next perception/navigation investigation should examine whether the
observer can acquire the needed floor area before approach, using prospective
view selection and a deployment-valid sensor pose. Any eventual sensing or
recovery change must be shared across matched baselines and evaluated with
fresh closed-loop executions. Pixel interpolation, blanket foot-contact
waivers and footprint shrinkage are not supported by this result.

The diagnostic source is retained as a read-only component, not adopted into
navigation. Tests verified closed tiling, outward enclosure, fixed bounds,
and rejection of even one uncovered tile: 2 passed in 0.13 s. The complete
artifact/source/model revalidation passed after replay. Post-admission replay
and diagnosis took 47.582701198989525 s in one CPU process.

Artifact root: `go2_training_bias_floor_coverage_v1_attempt_001` under the fixed
navigation development artifact root.

- Launch: `78525000cbaf7b2a23aba027392b5bf8a41fdaf88d9ec94ab9f3bc80cd44988d`
- Coverage witnesses: `c114a96cad0dcc0cbc9b76e8b9d41150cfb47a82e72b0f9d3c2093e3b15ec817`
- Result: `e1bf8b60a49482bd43975f6163a767d007d8d78d439f66d8f9a78c519454b863`
- Input probe: `5e48672a074d2086d734d02844af499d49d0a57571b9148b433b65ad2bedcfb5`
- Input readout: `f5ad4ad07dd06db5783143d044216b5371d7703fcdd3a692962f40cde887636f`

Zero verified arrivals and zero independent novel-maze evaluations remain.
Real-time execution, physical backtracking, matched reactive/nonpredictive and
planning/memory comparisons, realistic sensing and bounded hardware evidence
are still required. The active overall goal remains incomplete.
