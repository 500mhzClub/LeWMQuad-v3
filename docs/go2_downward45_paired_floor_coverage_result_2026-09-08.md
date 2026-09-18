# Actual 45-degree floor-coverage diagnosis result

All 63 recorded JEPA decisions reproduced exactly with unchanged corrected model
state and controller. At the first wait (42) and terminal (52), subdividing the
original complete 44-mm foot squares and combining all retained primary and
auxiliary observations still failed every one of the twelve candidate queries.
No aggregation change is justified by this result.

At the finest tested 8-by-8 subdivision, primary coverage was zero for every
query. Joint coverage equalled auxiliary coverage:

| Candidate | Covered tiles at tick 42 / 64 | Covered tiles at tick 52 / 64 |
| --- | --- | --- |
| hold | 40 | 23 |
| forward | 50 | 40 |
| left_arc | 50 | 39 |
| right_arc | 45 | 33 |
| left_turn | 41 | 24 |
| right_turn | 36 | 19 |

The hold-square centres were [0.2825724713412288, -0.3712047061633772] m at 42
and [0.2779807981583678, -0.3624955946061086] m at 52. Every uncovered hold tile
(24 and 41 respectively) was NEVER_ENTIRELY_IN_FRUSTUM in the retained history.
None was a visible tile rejected only by floor-pixel classification. These are
actual new-trajectory gaps, despite the earlier capture covering the different
30-degree predecessor's diagnosed coordinates.

This is a retained negative diagnosis. It supports neither filling unseen floor
nor inferring an unexecuted goal outcome. It also does not justify selecting
another camera angle solely to cover these new coordinates. The full native
45-degree pair remains 0/2 verified arrivals, with all measurement gates passing.

Next design review should separate two questions before another native attempt:
which measured contacts count as permissible foot/ground contact, and what
observation coverage is needed to claim support or clearance. The current code
permits a no-measured-intersection candidate with unobserved foot projection,
but requires full projection coverage to exempt measured floor intersections.
That is a conservative policy choice, not a demonstrated support certificate;
review its consequences explicitly rather than silently relaxing a tolerance.
Any successor must retain non-floor/unknown and non-foot obstacle evidence,
state its changed contact/support semantics, and preserve these failures.
If complete support coverage remains required, evaluate the full body-relative
foot/motion envelope and self-visibility prospectively, rather than fitting
sensor geometry to one failed trajectory. Direct-model nominal clearance and
the approximately 0.8-second full-loop time remain separate unresolved issues.

The diagnostic bound 1,302 sources, used one CPU process/thread, and completed
in 42.747171296039596 seconds after launch. It made no native commands or
controller changes and provides no independent-maze or hardware qualification.

Root: `go2_downward45_paired_floor_coverage_v1_attempt_001`.

| Artifact | SHA-256 |
| --- | --- |
| launch.json | 8afa714a31e69b13f349178bb986cfd217605176653e14b65c0469943ed37c27 |
| coverage.json | 28041d76f031d95b406b0a1180ad7679b091297baea13755c606e0138a548b0e |
| result.json | cbb2e078b0d4c6473640f2dbeaa633ab5f4d241a3e394c0d917ed280118dafb0 |
