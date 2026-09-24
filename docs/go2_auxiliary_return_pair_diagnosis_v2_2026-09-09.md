# Same auxiliary pair diagnosis with corrected guarded artifact path

The V1 attempt stopped at its first auxiliary PNG open, before pair fitting:
artifact_path was given the episode subdirectory instead of the attempt root.
Preserve that failed attempt and frozen sources. Its launch is
00438ea2df4021ae4c08343dea2aa5a78fe1848065a46e93712a90306f2f3e1e;
failure fc7b88723e3fda3a2dfa302283dd4a6df6c81c898523b71781985b197b7a24e3.

Run the separately named V2 script/root, changing only the auxiliary PNG path
adapter to artifact_path(INPUT, CASE + relative_png_name), adding exact failed-
attempt bindings and new output/protocol/status identities. Retain the same18
frames,25 pairs, source adapter, detector, matcher, fitting thresholds, fixed
calibration conversion and postfit-only native error evaluation described in
go2_auxiliary_return_pair_diagnosis_v1_2026-09-09.md. No scientific rule, sensor
array, model, running experiment or old outcome changes. Inherit and validate
the V1 frozen sources as well as all original inputs. Destination:
go2_auxiliary_return_pair_diagnosis_v2_attempt_001.

Check the corrected exact path before launch, run --preflight-only, then run
once if source/input/resource admissions pass. Same one CPU process/numerical
thread,2GiB available RAM and64MiB output above40GiB reserve beside the existing
native scene and independent replay. No native execution, training, observer
installation, online continuity, uncertainty, navigation or hardware claim.
