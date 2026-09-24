from copy import deepcopy
import json
import pytest
from lewm.recorded_visual_evidence_identity_development import restore_identity


def test_only_identity_type_restored_without_mutation_or_numeric_change():
    recorded={'identity':[0,0,0],'current_pose':{'position_initial_body_m':[.1,-.2,.3]}}
    before=deepcopy(recorded);restored=restore_identity(recorded,(0,0,0))
    assert restored['identity']==(0,0,0) and isinstance(restored['identity'],tuple)
    assert json.loads(json.dumps(restored))==before and recorded==before
    restored['current_pose']['position_initial_body_m'][0]=9.
    assert recorded==before


@pytest.mark.parametrize('identity',[[0,1,0],[False,0,0],[0,0],[0.,0,0],(0,0,0)])
def test_reject_mismatched_or_undeclared_json_identity(identity):
    with pytest.raises(ValueError):restore_identity({'identity':identity},(0,0,0))
