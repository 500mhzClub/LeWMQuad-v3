# Measured floor-extent rejection reconstructed

Completed go2_dual_camera_floor_extent_diagnosis_v1_attempt_001, session25318
exit0. Launch edd9d0d591af674629abf36b52c44a6451fe7c4f85bdd332bdccc0047cf6f9c9;
result1677bd7be89e5dd85c2e467693575aa5144f3f72fcef09ab16e8afa38b513a35.
All1618 source bindings and65 selected closed eleventh-collection inputs passed
before/after checks. Wall61.568442032s. One CPU thread, no scene/model inference.
Preflight42731 passed with73,620,197,376bytes available RAM and104,776,839,168bytes
artifact space,3.4%CPUbusy. The independent native raw audit remained active.

All14 selected frames reconstructed:0,1868,1872,1894..1904. Every earlier
admitted joint plane exactly matched its saved receipt. The first floor failure
is1904, confirmed by examining all preceding terminal fields. Original moments,
candidate selection, fitting, all-point admission and thresholds are unchanged.

| Frame | Auxiliary candidates | Second eigenvalue (m²) | Second-axis standard deviation |
| --- | ---: | ---: | ---: |
| 1894 | 8573 | 0.007020494301 | 83.79mm |
| 1902 | 6894 | 0.002942990768 | 54.25mm |
| 1903 | 6780 | 0.002648481588 | 51.4634mm |
| 1904 | 6514 | 0.002327368543 | 48.2428mm |

The fixed criterion is0.0025m², equivalent to50mm spatial standard deviation.
The patch progressively narrows and crosses that criterion at1904. This is
not loss of every floor point or an observed arithmetic composition mismatch.
At1903 and1904 primary candidate count is zero; current auxiliary visual pose
witnesses still pass the original accessor. Full fresh visual replay remains
the native audit's responsibility.

The rejected1904 algebraic plane has maximum all-point residual3.291055885e-6m
and RMS1.096196045e-6m. These are diagnostic statistics of an unadmitted fit,
not grounds for lowering the extent gate. After fitting only, native pose
diagnostics put its candidate world heights between-2.886643457e-6m and
5.771590678e-6m, with no robot-labeled candidate. Native floor-normal angle is
1.039902493e-5rad. Native poses/segmentation did not enter candidate selection,
fitting or any controller decision. All candidate points/pixels are saved.

The diagnosis itself establishes no successful return or uncertainty bound.
Before prospective use, all65 selected input bindings must match the final
independently audited eleventh artifact map. The separate measured-floor
transport successor retains the unavailable-plane evidence and original gates;
its source/protocol is docs/go2_measured_floor_transport_prefix_v1_2026-09-09.md.
