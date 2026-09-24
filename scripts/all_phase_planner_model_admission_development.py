"""Original expanded admission and fresh byte-identical planner-compatible copy."""
from lewm.all_phase_planner_model_adapter_development import AllPhasePlannerModel
from scripts.all_phase_translation_bias_model_admission_development import admit, load_assigned as original_load


def load_assigned(admission, name):
    source, condition, variant = original_load(admission, name)
    return AllPhasePlannerModel(source), condition, variant
