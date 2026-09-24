# Auxiliary raw depth prefix V1 terminal acquisition failure

Attempt `go2_auxiliary_tilted_depth_prefix_v1_attempt_001` failed during the first
auxiliary frame's diagnostic segmentation-map parsing. The native renderer
stores background as `0: -1`; its nonbackground entries contain entity/link
tuples. The capture helper incorrectly attempted to iterate the integer
background value, raising `TypeError("'int' object is not iterable")`.

The scene was built with robot visuals enabled and completed settling and the
initial primary capture. Zero commanded-prefix ticks were executed and zero
auxiliary frame receipts were completed. The primary transform restoration
completed without another exception; finalization persisted the initial raw
physical and primary sensor artifacts. This failed acquisition supplies no
usable auxiliary-coverage result or navigation evidence. Native Genesis also
emitted its existing neutral-position joint-limit warning; that was not the
terminal exception.

The original source and attempt remain immutable. A distinct integrity
successor will validate background `0 -> -1` separately from every nonnegative
entity/link tuple, retain background in the metadata, and exclude it from robot
labels. That successor must preserve the camera, renderer, physics, command
prefix, source/input audit and terminal conditions exactly. No camera or
scientific-selection change is justified by this parser failure.

- Launch: `1b05b3e771f29b54f79431b8a4af6f800b590c173a3b59a5ea9a0c84c69e50ca`
- Failure: `df0defa5c20888f67c077faa34cc6b0ddc2c2acc88434be01deb196c57fcb1e4`
- Empty auxiliary receipt and command-tape SHA256:
  `37517e5f3dc66819f61f5a7bb8ace1921282415f10551d2defa5c3eb0985b570`
- Initial primary camera audit:
  `c428110796323be2cc95e416135ac9a705371bbc48e4fcbe2c88c45c5c9ec1fe`

Before launch, a syntax error in the new scope test was corrected; the two
scope tests then passed in 1.62 s. That test collection failure did not launch
a native scene. The later native failure is separately preserved as above.
