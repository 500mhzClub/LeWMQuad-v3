# Eligible-cell floor-registration replay

Compare two fresh floor-registration objects over observations 0 through 853
of the completed original no-RGB JEPA direct-flow tracking simulation. Both
receive the same original public primary and auxiliary depth packets and the
same recorded original visual pose evidence. The sole substitution is the
eligible-cell floor-index function inside measured-candidate extraction.
Private function bindings preserve the original function bodies and imported
module globals. All original plane fitting, registration, transport, reference,
anchor, clock, validity and failure checks remain unchanged.

Every complete registration result must equal the original recorded floor
evidence. Both complete object states must match after each observation. Public
input fingerprints must remain unchanged. Paired execution order alternates by
frame; timings exclude input reconstruction and receipt/state comparison.
Timing is observational on the shared host and does not establish controller
speed or real-time operation.

The dense floor-index calculation is retained for ambiguous numerical boundary
decisions. No scientific threshold is widened. Synthetic tests and the fixed
microbenchmark precede this real-input replay. The V1 benchmark preparation
failure is preserved; V2 supplies the missing source-discovery argument and
completed all 100 paired kernel comparisons with identical output bytes.

This replay uses the authenticated completed original worker and all its
artifact bindings. It verifies sources and input bindings before and after
execution, preserves terminal failures, creates one exclusive output, and
consumes no observation after 853. It loads no model, reexecutes no visual
observer, runs no policy or physics, and makes no navigation claim. Mapping
integration and complete controller replay remain necessary before adopting
this performance change in a future simulation.
