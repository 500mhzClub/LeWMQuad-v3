# Unchanged tracking fix recovers the maze3 failure in full-controller replay

Session47935 exited0. Root:go2_direct_flow_maze03_prefix_v1_attempt_001.
Result104edd70824e7d4e7b909be924eb68a918542efd3464ff3247f9570ddc28669e;
launch362e4be722723b0742224e7f99d13ffbda97c567c2d611bccd947a16d793fce0;
decision stream5525a9ddeaae3c6a30865bdd3cc21bb98f8d9644cb2a9e00f8171f94c8d49aa0.
All1,677source/input/artifact bindings completed,429.1678464s wall time. Result
and output hashes were checked again when reading these findings.

All264 preceding complete original decisions and actual commands match, with
261 prior raw forecast banks compared. At original failure observation264, the
unchanged direct-flow implementation admits a current primary-camera anchor
measurement from reference263. It uses13 valid matched depth pairs; the same
rigid and temporal checks qualify the measurement. No association threshold,
bridge budget, reference/pose history, controller, mapper, model or mission
change was introduced relative to the completed maze1 tracking replay.

The full controller advances through264 with current visual/registered pose,
mapping and forecast selection, no failure or terminal, and requests left turn
[0,0,0.45] instead of the original terminal zero. Both live sensor contracts pass
before serialization. Model state remains
4f796afdf32c2c6dbcd1d5981834a7b17be86e2efdafd9427b2d80240def0ca6;
weights and public inputs unchanged, gradients absent. No observation265 or
changed-command physical outcome was consumed or inferred.

This is now a complete recovery prefix on the other previously observed
development visual-failure case, alongside maze1. It does not establish fresh
closed-loop continuation, an arrival or a round trip. A future native maze3
experiment must reproduce13,950physics samples,265paired observations,264prior
commands,261prior forecast banks and all265prospective candidate decisions before
comparing new physical outcomes. Such a native launcher is not yet implemented.
The existing native queue remains residual maze2, tracking maze1, supervised1–3.
