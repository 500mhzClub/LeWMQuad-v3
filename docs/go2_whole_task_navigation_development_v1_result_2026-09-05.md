# Whole-task V1: terminal infrastructure failure before navigation

Collector72844 exited1, with zero of four planned trials completed. The first
scene was built, but the first 2-ms settling sample raised `KeyError('source_node')`
in the inherited native logger. No controller decision, beacon discovery or
return occurred. The intended four-trial memory comparison is untested.

The scene constructor correctly accepted only wall boxes and spawn. Review had
missed a second dependency: the inherited `_sample` computes legacy source,
target and directed-edge polygon memberships, and the settling command executor
aggregates those fields. Whole-maze scenes deliberately have no such privileged
route annotation. Mock controller, static-object and constructor-wiring tests
did not exercise this native sampling path;1,017 passing tests did not establish
collector compatibility. All failure artifacts and237 bound source paths remain
unchanged. This run must not be restarted or silently repaired.

Original launch SHA-256:
`1acf0ba6d0d5323365463d0665c89f508ec8eccdbf98ba21d3dfcea66aa3d272`.
Original terminal result SHA-256:
`49abcdc09641c167aee11054f04fb6a7a19526afc8b0a6deb623acc84e2a2c25`.

The next explicitly separate correction removes only unused region annotation
and aggregates actual raw trace fields. It must preserve native state readback,
contact/body stops, gait execution, sensor clocks, controller, scene geometry,
memory comparison and physical success criteria. A method-resolution test must
show that ordinary/fast sensor and contact-stop wrappers are still invoked.
Do not inject dummy region labels merely to satisfy the old logger. The original
study remains terminal; a new root and binding identify any correction attempt.
