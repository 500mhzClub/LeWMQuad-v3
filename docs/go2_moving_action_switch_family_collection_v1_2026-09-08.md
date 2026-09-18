# Moving-action switch family collection V1

Collect exactly the 144 prospectively assigned cells in
`lewm/moving_action_switch_family_development.py`, in its fixed seed-2026091301
opaque order. The design document supplies the four canonical existing geometry
identities and the full prefix/suffix/zero-drain schedule. This is one new
development population, 72 training cells and 72 reused-cluster geometry-transfer
cells, with no independent novel-maze or final benchmark evaluation.

Admit the exact completed scaling result and all its phase artifact bindings.
Require collection sources to equal the benchmarked source manifest. Use only
the selected one or four fresh CPU workers, one episode per process, bounded
batches and single OpenCV/BLAS threads. Check 32-GiB available RAM, the measured
storage estimate against the fixed 24-GiB allowance, and that allowance above
the unchanged 40-GiB free artifact reserve. Record hardware and monitor resources.

Each episode independently initializes its canonical scene and executes three
quiet ticks, ten moving-prefix ticks, forty suffix ticks and ten zero-drain
ticks after the unchanged native settle. Frame 13 is the branch; only frames
10 through 13 enter its intended past context. Commands use exact existing
candidate plans and the unchanged actuator. No model, observer estimate or
native evaluator state selects these prospective commands. The old gyro
observer is recorded as a shadow and exactly replayed, with no availability
requirement for executing the fixed schedule.

Persist and verify every raw sensor, native contact, physical geometry,
setup/friction/gain, ordered raster, command tape and terminal witness. Run the
unchanged numerical near-field, strict visibility, stable-interior and
near-occlusion checks. A worker/infrastructure, acquisition or measurement
failure stops additional batches; running siblings finish and preserve their
receipts. Expected physical stops remain recorded cells with censored motion
and image labels. No retry, replacement, prefix-state copy or resume is allowed.

Audit the complete 144-cell population before learning. For each cluster and
prefix, all six suffix siblings must match exact raw physical arrays and public
sensor packets, RGB/native-depth pixels and camera poses through frame 13.
Mixed availability or any mismatch fails the prefix gate. If all six siblings
stop before an available branch, retain all six as unavailable in the full
denominator. Do not filter unfavorable outcomes or train a partial population.

Derive target-only native branch-body XY and projected relative yaw at eight
500-ms horizons. Disallowed native contact is absorbing across those horizons;
motion/images after contact are invalid. A stopped execution without an
observed endpoint or contact remains unknown, not a zero target. Publish
role/cluster/action counts, full and unavailable branches, physical stops,
valid/contact/censored target counts and every prefix comparison.

Root: `go2_moving_action_switch_family_v1_attempt_001` under the owned navigation
artifact base. A passing data collection permits development of a separately
frozen learning study; it supplies no new trained model or navigation result.
Multiple optimization seeds, full/no-RGB and direct/supervised-rollout/JEPA
comparisons remain required. All earlier failures and the full end-to-end
novel-maze navigation goal remain in force.
