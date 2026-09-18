# Fixed subpixel feature comparison on the diagnosed return transition

Compare the original and separately named subpixel feature extractor on all
25 pairs in the completed return-transition match diagnosis. This is a
post-hoc development comparison on reused public observations, not a prospective
navigation result. Freeze one candidate: cornerSubPix radius 5 pixels, 30
iterations, 0.01 pixel stopping precision; reject nonfinite refinements,
displacements over 2 pixels, duplicate half-pixel locations and invalid or
discontinuous depth. Recompute upright size-8 SIFT descriptors at measured
refined coordinates. Preserve all original mutual ratio, optical-flow,
12-match, six-cell, rigid consensus and gyro-consistency rules.

Use exactly the previously bound frames 1853 through 1870. Integrate their
public fast gyro from a local identity at 1853, exclusively to provide the
reference-relative rotation consistency gate. This local diagnostic coordinate
origin is not an online pose reset. Fit each pair independently using the
original joint register function and actual current frame seed; report every
success and failure. Pair fits do not establish continuity, global accuracy,
calibrated uncertainty or suitability for deployment. No candidate is installed
in the running audit or original controller. No parameter search, model load,
training, simulator or new sensor acquisition.

Bind the completed diagnosis result, inherited launch/input/source bindings,
new source/test/protocol bindings and every consumed artifact before and after.
One CPU process and one OpenCV/numerical thread; inspect hardware immediately
before the 18-frame task. Admit 2 GiB RAM and 64 MiB output above the 40 GiB
storage reserve. Preserve the exclusive output
go2_return_subpixel_pair_comparison_v1_attempt_001, including any failures.
