# Downward RGB-D support on the ninth run's failed return-transition pairs

Diagnose the same18 frames1853..1870 and25 pair definitions already fixed by
the completed primary-camera diagnosis:17 consecutive pairs and eight primary
retained-reference frame indices paired with1870. Those eight indices define
analysis pairs, not a claim that an auxiliary online tracker retained them.
Reproduce all original primary feature witnesses and correspondence stage
counts exactly. Add the simultaneously recorded downward RGB images and their
validated auxiliary depth, using the unchanged corner/upright-SIFT detector,
ratio/mutual/LK/depth gates and joint rigid fitting rules. No parameter search
or threshold change. Preserve every pair, including rejections.

Validate auxiliary PNG pixel hashes against acquisition receipts and reconstruct
the exact auxiliary depth packet against its paired primary identity/time and
fixed calibration. The current controller consumes auxiliary depth but not this
auxiliary RGB channel. This diagnostic does not fabricate an existing RGB
packet contract or install a second-camera observer.

Existing lifting/projecting helpers use the primary fixed extrinsic. For
auxiliary images their output is explicitly a reference coordinate frame,
not robot-body coordinates. Use the fixed auxiliary-from-primary calibration
adapter: conjugate relative gyro rotation into that reference frame before
fitting, then convert fitted rotation/translation back, including the camera
lever arm. Five synthetic tests passed, including physical-point and rigid-pose
equivalence and invalid input rejection. The fitter's existing reference-frame
limits remain unchanged; report converted body translation/rotation envelope
checks separately. Pair fitting is not continuous online pose acceptance.

All feature extraction, matching and fits use public images/depth and gyro.
Only after every fit is complete, read the bound native physics poses at the
corresponding observation indices to compute independent pair rotation and
translation errors. Native poses never enter matching, fitting, calibration,
pair selection or controller commands. This is a retrospective diagnostic on
reused development data, not independent-layout or prospective recovery evidence.

Bind the completed ninth native result, original primary diagnosis result/
launch and its exact public inputs, the auxiliary RGB/depth files and acquisition
receipts, physics trace used solely for postfit evaluation, all new sources and
adapter tests. Revalidate before and after. Original failure and visibility
outcomes remain unchanged. Exclusive output:
go2_auxiliary_return_pair_diagnosis_v1_attempt_001.

Run --preflight-only and inspect topology/affinity, CPU/GPU/VRAM, RAM, competing
work and both volumes. One bounded CPU diagnosis/one numerical thread beside
the existing native scene and independent controller replay. Require2GiB RAM
and64MiB output above40GiB reserve; capacity admission is not an OS resource
limit. No training, new native execution, running-source mutation, candidate
installation, online-continuity, uncertainty, navigation or hardware claim.
