"""Fixed paired development layouts and one authenticated supervised rollout fit."""
from copy import deepcopy
from lewm.independent_floor_transport_study_development import LAYOUTS
from lewm.matched_rollout_objective_admission_development import NAMES
from lewm.independent_reactive_floor_transport_study_development import MATCHED_KEYS, OUTCOME_KEYS

SUPERVISED_STATE='171c8576d2c3fcfd0ce698351acf86a05f2ea29bdf829041e98641cdac778a73'


def planned_cases():
    return [(f'full_supervised_rollout_novel_maze_{i:02d}',i,'full','supervised_rollout',NAMES[1]) for i in LAYOUTS]


def paired_outcomes(learned,supervised):
    if len(learned)!=3 or len(supervised)!=3:
        raise ValueError('both complete fixed three-layout cohorts required')
    pairs=[]
    for index,old,new,case in zip(LAYOUTS,learned,supervised,planned_cases(),strict=True):
        if (type(old['layout_index']) is not int or type(new['layout_index']) is not int
                or old['layout_index']!=index or new['layout_index']!=index
                or old['case']!=f'full_jepa_novel_maze_{index:02d}' or new['case']!=case[0]
                or old['status']!='INDEPENDENT_FLOOR_TRANSPORT_COLLECTED_AND_RAW_AUDITED'
                or new['status']!='SUPERVISED_ROLLOUT_COLLECTED_AND_RAW_AUDITED'
                or new['model_state_sha256']!=SUPERVISED_STATE or new['model_state_unchanged'] is not True
                or new['prefix_comparison']['complete_candidate_decisions_match_prospective_prefix'] is not True):
            raise ValueError('ordered complete paired native results with actual intervention required')
        pairs.append(dict(layout_index=index,jepa=deepcopy({k:old[k] for k in OUTCOME_KEYS}),
            supervised_rollout=deepcopy({k:new[k] for k in OUTCOME_KEYS})))
    return pairs
