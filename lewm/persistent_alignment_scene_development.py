"""Same development fixtures, separately identified persistent-feedback panel."""
import copy

from lewm.initially_aligned_scene_development import trials as predecessor_trials


def trials():
    rows = copy.deepcopy(predecessor_trials())
    for spec in rows:
        spec.update(scene_id=f"persistent-alignment-development-v1-{spec['case_index']:02d}-{spec['method']}",
                    family='PERSISTENT_ALIGNMENT_CONTINUATION_DEVELOPMENT',
                    procedural_seed=2026093000+spec['case_index'])
    return rows
