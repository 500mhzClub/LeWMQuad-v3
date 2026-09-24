# Ninth maze arrival and return-transition evidence

The ninth collector has stopped, with 1880 commands, 1881 paired observations,
94750 physics samples and ten zero-command terminal-drain ticks. It first
reported an outbound arrival at frame 1866, switched to RETURN, then failed
visual tracking at 1870. The full native raw-sensor/command audit and prefix
comparison remain running in session59231 at this report's creation. This
report does not replace those checks or claim a verified arrival/round trip.

Input root: go2_later_floor_resolution_maze_pilot_v1_attempt_001, case
full_jepa_novel_maze_00. Launch SHA-256
c3d035abcc69b3b42ecb160021203e7d6d685a176e860e5c3044afa2035cefa4.
Collection result69532d7591532ba5400e05a6cf596766df567a660392d973f4ef22da52845a15;
physics tracece14c1c1cad3dccb3db81bca554a12142f7dbe247ec39c32473153df3c17fc49;
decision stream23510c874165007e055cbd1fc49346df38fbad3b30b9d96ea68f22fd3f536aa9.

## Arrival dynamics: braking included in the quiet window

The original numerical evaluator returned an outbound window maximum goal
distance of0.037619440722m and maximum3D speed0.152474489415m/s. Its one-second
arrival-and-quiet check is false against the unchanged0.05m/s threshold.
The six-edge loop-erased outbound route was traversed with no invalid edge
crossings; there was no physical return crossing. These numerical observations
remain contingent on the full raw audit, and strict visibility is unresolved.

Separate fixed-population diagnosis92614 CLOSED exit0:
go2_ninth_arrival_dynamics_v1_attempt_001, result
c0cfd9ab1d40c7497735f499dfb8d9ea5e8717f028c165e20882ce835ed88227,
launch e9e7ca875ba186fbfc283b2b727ca865c8ed4cd628eb9876412f591f36d6576c,
1542 sources. All31 windows ending at frames1850..1880 were recorded; none
passes the combined one-second distance, speed and zero-request requirement.
All inputs and sources were revalidated after processing.

At claimed arrival1866 the current speed was0.002094342441m/s, but the
one-second window began at sample93549 with speed0.152474489415m/s. The first
zero-request interval ending1857 still reached0.145616155248m/s. The next
interval ending1858 stayed below0.05m/s, and the robot slowed further.
At1868 return-turn requests had begun, so no later complete zero-request
window repairs the original arrival claim. The mission currently counts
proximity plus zero requests, without a measured-motion gate. A successor
should begin its one-second observed dwell only after measured settling;
sampled visual displacement alone would still not certify continuous native
speed or calibrated uncertainty. Do not simply relabel the original arrival.

## Visual tracking: match-location checks remove needed support

Diagnosis34677 CLOSED exit0:
go2_return_transition_match_diagnosis_v1_attempt_001, result
29565cd5f2d87ba6c9d657780fc9ca75f908ab7cd28d0713f4da28d2b758e081,
launch e62e940f7cdd00249d81f9c2f7f5476d669450b50910331c2eb90500d98d3a07,
1542 sources. It reconstructed18 public RGB-D frames1853..1870 and all25
declared pairs. Every diagnostic correspondence array equals the original
matched_points output; all source/input bindings passed before and after.
Resource admission recorded16 physical/32 logical CPUs,3.6% aggregate CPU,
75,735,244,800 available RAM bytes and118,645,006,336 artifact-free bytes.
Only one bounded CPU diagnostic ran beside the existing native audit.

Frame1870 has22 detected/liftable/selected corners, not the29 from the last
accepted frame1869 shown in the failure snapshot. Pair1869->1870 has12 mutual
unique descriptor matches; all12 pass finite flow and forward/backward checks,
but only8 pass the unchanged one-pixel independently detected feature-location
check. Depth lifting removes none. The eight retained references produce
7..11 final correspondences. All nine fail the original minimum12-match
requirement before rigid fitting. The saved RGB image shows a close view of
large low-texture checkerboard wall patches; that visual observation does not
establish a calibrated cause or justify weakening the pose gates.

## Fixed subpixel candidate: failed terminal support, not adopted

Added the separate SubpixelCornerSupportFeatureFrame and four focused tests.
All four passed0.17s, covering known corner localization, input nonmutation,
invalid-depth rejection and blank images. The fixed candidate retains the
original downstream matcher and joint registration thresholds.

Comparison22220 CLOSED exit0:
go2_return_subpixel_pair_comparison_v1_attempt_001, result
e1989bb27999ff04100700a121c6dea3bef81ed22a041afc05474e00eb38ca23,
launch9c55a588b2037f20bea0524fb3f396272b90b09a835c86852502cc8190934d8f,
1546 sources. All25 original and candidate pair outcomes are saved. Original
stage counts reproduce exactly; source/input identities pass before and after.

The candidate passes rigid fitting on all16 preterminal consecutive pairs
(the original fails one of those pair fits, which does not itself imply an
online failure because retained anchors also exist). However, all nine pairs
to1870 still fail. The current refined detector retains11 features after
rejecting11 refinements/deduplications; maximum attempted refinement is
2.722764730453491 pixels against the fixed2-pixel displacement cap. The
consecutive pair improves from8 to10 matches, still below12. This is a failed
candidate for the terminal issue; it was not integrated into a controller and
no new native run was launched. Preserve its exact implementation/result.

## Remaining work

Follow native59231 through its full audit and comparison, then run the prepared
completed-native readout with the actual terminal result hash. Diagnose robust
feature availability during the return turn and implement measured settling
in a separately named successor. The full goal remains active: no verified
round trip, independent-layout successes, matched baseline/ablation evidence,
real-time operation, calibrated sensing or bounded hardware validation yet.
