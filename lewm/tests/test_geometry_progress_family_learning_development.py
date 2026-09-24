"""Same causal tensors and masks with exact new prospective assignments."""
from copy import deepcopy
from pathlib import Path
import ast
import pytest
from lewm.geometry_progress_layout_family_development import assignments
from lewm.geometry_progress_family_learning_sample_development import materialize
from lewm.tests.test_geometry_progress_learning_sample_development import fixture


def new_fixture():
    reader,row,calls=fixture();trial=next(iter(assignments()))
    row.update(trial=trial,**assignments()[trial]);return reader,row,calls


def test_same_tensorization_code_with_new_assignment_authority_only():
    a=Path('lewm/geometry_progress_family_learning_sample_development.py').read_text().replace(
        'geometry_progress_layout_family_development','geometry_progress_pilot_development')
    b=Path('lewm/geometry_progress_learning_sample_development.py').read_text()
    ta,tb=ast.parse(a),ast.parse(b);ta.body.pop(0);tb.body.pop(0)
    assert ast.dump(ta)==ast.dump(tb)


def test_new_context_preserves_all_masks_and_never_reads_missing_futures():
    reader,row,calls=new_fixture();s=materialize(reader,row)
    assert calls==[0,1,2,3,8]
    assert s['targets']['contact_valid'].all() and s['targets']['contact'].sum()==6
    assert s['targets']['motion'][2:].isnan().all() and s['targets']['future_valid'].sum()==1
    assert set(s['inputs'])=={'observation_history','known_action_blocks','known_action_valid'}


@pytest.mark.parametrize('field',['geometry','cluster','data_role','opening','action','appearance_seed'])
def test_assignment_mutations_rejected_before_packet_access(field):
    reader,row,calls=new_fixture();row[field]='changed'
    with pytest.raises(ValueError,match='assignment'):materialize(reader,row)
    assert calls==[]


def test_legacy_pilot_assignment_is_not_relabelled_as_new_family():
    reader,row,calls=fixture()
    with pytest.raises(ValueError):materialize(reader,row)
    assert calls==[]
