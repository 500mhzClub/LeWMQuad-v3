# Mesa 25.2.8 subpixel source review

Installed package queries report libgl1-mesa-dri, libegl-mesa0 and libglx-mesa0
version 25.2.8-0ubuntu0.24.04.2. The prior native context result
fc2d51f3011294573247cfb1782f9c0631dca8a1daa1a8ae390af2e70db60819
reports llvmpipe LLVM 20.1.2, 256 bits, Mesa 25.2.8 and OpenGL 4.5 core.

Downloaded the [official upstream source archive](https://archive.mesa3d.org/mesa-25.2.8.tar.xz),
43,813,260 bytes, SHA-256
`097842f3e49d996868b38688db87b006f7d4541e93ce86d2f341d8b3e7be7c93`.
It matches the hash in the [official release notes](https://docs.mesa3d.org/relnotes/25.2.8.html).
Only three explicitly named source files were extracted to
`/tmp/lewm_mesa25_precision_mw3x54ji`; nothing was built, installed or substituted
in the running experiment. These files are under src/gallium/drivers/llvmpipe:

| File | Bytes | SHA-256 |
| --- | ---: | --- |
| lp_setup_tri.c | 42208 | fabf85a1370e015b92c807d46f3cfeac44d4676fbd74e863073d4486c574dd3a |
| lp_rast.h | 11916 | 9317582052480b22886026e1e7de16af9af25e60de106cbdaa4262f52126b0f2 |
| lp_setup.h | 5312 | 5711a1406dad98dda3cdaac25caa605284d78c583dad0522b9604135568b8211 |

lp_rast.h lines 53–54 defines eight fractional bits and a scale of 256.
lp_setup_tri.c lines 1037–1099 subtracts the single-sample pixel offset,
scales float vertex coordinates, and converts to integers. The SSE path uses
_mm_cvtps_epi32; the alternative uses util_iround. The source explicitly notes
that tie rounding can differ between these paths. Thus upstream implementation
evidence supports finite-grid vertex snapping, but not every assumption in a
particular NumPy rounding reconstruction.

Remaining limits: the Ubuntu-patched installed binary has not been proved
equivalent to this upstream implementation; native floating-point rounding mode,
shader arithmetic, clipping and depth interpolation were not captured by this
review. Real wall visuals are subdivided at a maximum 0.125 m patch spacing,
so snapping original box-edge endpoints is not the same as snapping the actual
rendered triangle vertices. Neither eight subpixel bits nor this source review
alone proves a complete ideal-camera-to-rendered-depth error bound.
