from copy import deepcopy
import pytest
from scripts.read_go2_training_translation_bias_v1 import assert_only_position_score_changed


def test_position_changes_preserve_every_other_score_and_denominator():
    original=dict(role='train',samples=3,clusters=[dict(position_error_m=.02,yaw_error_rad=.1,
        motion_targets=3,contact_targets=4,contact_positives=1,undefined_yaw=0,contact_brier=.01)])
    corrected=deepcopy(original);corrected['clusters'][0]['position_error_m']=.005
    assert_only_position_score_changed(original,corrected)
    for key,value in (('yaw_error_rad',.11),('motion_targets',2),('contact_brier',.02),('undefined_yaw',1)):
        bad=deepcopy(corrected);bad['clusters'][0][key]=value
        with pytest.raises(ValueError):assert_only_position_score_changed(original,bad)
