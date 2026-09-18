# Paired history cannot fill the observed near-field blind region

All 110 JEPA decisions from the bounded reobservation mission replayed exactly
with unchanged model state. The diagnostic examined every candidate's front-left
foot at ticks 62 and 99, reconstructing the original whole-foot witnesses exactly.
Tick 62 is the first all-candidate auxiliary-floor stop, rather than a claim
about the first individual candidate veto anywhere in the mission.

No candidate's entire 44-mm footprint was covered by primary, auxiliary or
combined retained patches, at any of the tested 1/2/4/8 divisions per axis.
At eight divisions, primary covered zero tiles for every foot. Combined coverage
equalled auxiliary coverage exactly; the primary camera added no missing area.

| Candidate | Covered tiles at 62, out of 64 | Covered tiles at 99, out of 64 |
| --- | ---: | ---: |
| Hold | 13 | 24 |
| Forward | 27 | 46 |
| Left arc | 21 | 35 |
| Right arc | 25 | 46 |
| Left turn | 10 | 17 |
| Right turn | 14 | 29 |

The hold-foot centres were [0.338907689, -0.458980615] m and
[0.344389092, -0.462970580] m in the observed map. All 51/40 failed hold tiles
had never been entirely inside either retained camera frustum. No failed hold
tile was visible but rejected by the floor-pixel checks. This is missing measured
coverage; joining histories or shrinking subdivisions does not create it.

The paired-view helper passed a five-test group covering closed tiling, full
coverage from two complementary views, retained provenance, unchanged source
histories, mismatched clocks and a shared unseen strip. It remains diagnostic
only and is not adopted by the controller, because it does not resolve the
observed failure.

Next evaluate one fixed steeper auxiliary camera orientation, 45 degrees down
with the same mount, intrinsics and depth interval. Test full-foot geometric
visibility before new raw rendering. This changes sensor geometry prospectively;
it does not establish actual pixels, absence of occlusion, hardware calibration
or navigation success. Preserve the 30-degree camera's negative result.

Root: `go2_paired_retained_floor_coverage_v1_attempt_001` under the guarded
development base. The run bound 1,266 sources and took 72.125272 seconds.

| Artifact | SHA-256 |
| --- | --- |
| launch.json | 51c3181293ed6b24c88a4865da793ebebf7e52d763c139d724018e1dd7dbbb5b |
| coverage.json | 7af79579ef9cc3e3833fe9356a1babf3cc3f17870c5c24ca7b4b92a7a549767d |
| result.json | 7f5546925b8303da8f414ef6c27eb9d40cdf869304d09262b92c978367d2b152 |
