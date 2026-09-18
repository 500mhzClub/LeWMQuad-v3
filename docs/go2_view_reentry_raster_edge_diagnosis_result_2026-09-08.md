# Completed view-reentry raster edge diagnosis

This evaluator-only diagnosis preserves the failed native result and strict
visibility criterion. It changes no sensor pixels, masks, poses, commands or
navigation outcome. No native scene or model was loaded.

Output: `go2_view_reentry_raster_edge_diagnosis_v1_attempt_001`.
Launch SHA-256: `bcb05b54b2e5159105a966c96875e8ec0afa39188c0892ffe77fab0ffb6f568e`.
Result SHA-256: `73d6077b81430ebe78518634568a2fe4f5295966d551ca258c0c53f0a1f0b9a1`.
The launch binds 1,464 sources and completed predecessor artifacts, verified
before and after diagnosis. Two focused synthetic geometry tests passed.

The original failed frame is exactly 909 and the single sampled bad pixel is
row 260, column 428. Native optical depth is 1.075383186340332 m; the exact
camera-centre ray reference is 2.544169195999424 m. Their surrounding 3x3 grids
cross the foreground/background wall silhouette. The nearest projected edge,
shared by walls novel_wall_3_2_1 and novel_wall_4_1_0, is only
0.000038491271380950855 pixels from the pixel centre. Hypothetically rounding
both projected endpoints to the nearest 1/256 pixel moves its signed line
distance from +0.00003849127137665553 to -0.0000914206215859396 pixels.
Thus that hypothetical rounding changes which side contains the sample.

Captured native precision reports eight subpixel bits and a 24-bit depth target.
The four reported RGB sample positions are not evidence that the separately
captured depth uses the same multisample pattern. OpenGL 4.6 chapter 14 specifies
half-integer fragment centres, and table 23.53 describes SUBPIXEL_BITS as screen
coordinate precision. These facts do not establish the particular endpoint
rounding rule assumed by this diagnostic or a complete camera/geometry error
bound. Reference: [official OpenGL 4.6 core specification](https://registry.khronos.org/OpenGL/specs/gl/glspec46.core.pdf).

All 4,240 stable-interior sampled rays pass the existing footprint diagnostic;
146 original compared samples are boundary-ambiguous, including the bad one.
This is supporting evidence for an edge-precision explanation, not a certified
boundary measurement. Strict visibility remains false, as does navigation
qualification. A prospective precision treatment or native raster benchmark
is still needed; the old failure cannot be relabelled by this result.
