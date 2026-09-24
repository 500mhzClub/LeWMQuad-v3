"""Separately scoped same-fixture initial-alignment development panel."""
import copy

from lewm.observed_continuation_scene_development import trials as predecessor_trials


def trials():
    rows = copy.deepcopy(predecessor_trials())
    for spec in rows:
        spec.update(scene_id=f"initially-aligned-development-v1-{spec['case_index']:02d}-{spec['method']}",
                    family='INITIALLY_ALIGNED_CONTINUATION_DEVELOPMENT',
                    procedural_seed=2026092900+spec['case_index'])
    return rows
