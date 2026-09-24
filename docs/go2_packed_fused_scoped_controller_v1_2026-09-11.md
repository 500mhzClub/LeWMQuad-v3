# Packed-owned indices composed with fused scoped receipt construction

The completed current profile identifies measured-sample insertion as a
remaining cost. The repository already contains PackedOwnedMeasuredSampleBoundsIndex:
one packed voxel grouping, batch bounds updates and independent storage for
each final bounds array. Its previous component benchmark, full 1,881-decision
replay and paired 256-observation controller benchmark are existing evidence;
they must not be repeated or presented as results for this new composition.

PackedFusedScopedController extends the prepared FusedScopedBatchedController.
After the original empty-state constructor, it replaces exactly eight empty
persistent MeasuredSampleBoundsIndex objects: primary all, auxiliary all,
primary floor/other, auxiliary floor/other and confirmed auxiliary floor/other.
The entire index population must have exact original types, distinct identities
and all four dictionaries empty before any replacement. Incompatible or partly
populated state is rejected without partial replacement.

Memory and map types, their alias, model, observer, localization, residual,
mission, history, selector, frame-local indices and inherited intersection
methods remain. Therefore the selector's exact supported-memory guard still
admits its scoped cache. The new controller adds one explicit output flag and
controller label, with a narrow public-result normalizer to the fused baseline.

The state checker retains all values and normalizes only the existing two
patch-store tags plus eight exact index-type tags. It rejects mixed, shared or
unexpected index types. It does not mask changed bounds, counts, witnesses,
latest frames, observed maps, residuals or history. Tests cover real public
primary/auxiliary observations, articulated footprint equality, both retained
patch stores, cache hits, independent public receipt ownership, failure stops,
and state-change detection, alongside the existing packed-index tests.

The current fused-receipt replay remains the sole full CPU replay. This
component preparation runs no model inference, new raw replay or native scene.
No integration into a running or frozen native experiment is authorized by
this preparation. Once the current replay completes, a separately prepared
paired replay must compare the full combined implementation on the original
raw history, with exact decisions and all retained-state checkpoints. Its
timing must be measured independently; benefits from older runs cannot be
added. This does not establish real-time control or navigation success.
