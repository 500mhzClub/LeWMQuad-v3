import pytest

from lewm.context_matched_coverage_development import context_table,matched_schedule,AugmentedCausalDataset


def rows():
    result=[]
    for layout in ('a','b'):
        for context in ('initial','moving'):
            for action in range(5):
                result.append({'window_id':f'{layout}-{context}-{action}','layout_id':layout,'data_role':'train',
                    'context_id':layout+'-'+context,'action_index':action,
                    'source_kind':'old' if context=='initial' or action==1 else 'switch','source_index':len(result)})
    return result


def test_schedule_matches_current_context_and_initial_choices_but_expands_moving_actions():
    source=rows(); schedule=matched_schedule(source,updates=50,seed=9); changed=0
    assert len(schedule)==50 and all([r['layout_id'] for r in s['batch']]==['a','b'] for s in schedule)
    for step in schedule:
        for pair in step['batch']:
            old=source[pair['coverage_limited']['dataset_index']]; new=source[pair['expanded']['dataset_index']]
            assert old['context_id']==new['context_id']==pair['context_id']
            if pair['context_id'].endswith('initial'): assert pair['coverage_limited']==pair['expanded']
            else:
                assert old['action_index']==1
                changed+=new['action_index']!=old['action_index']
    assert changed>0 and schedule==matched_schedule(source,updates=50,seed=9)


def test_no_new_conditioning_state_can_be_smuggled_in_with_new_action_data():
    source=rows(); source[6]['context_id']='unseen-context'
    with pytest.raises(ValueError,match='new current context'): context_table(source)


@pytest.mark.parametrize('fault',['window_id','action','role','layout','validation'])
def test_duplicate_or_cross_role_source_is_rejected(fault):
    source=rows()
    if fault=='window_id': source[1]['window_id']=source[0]['window_id']
    if fault=='action': source[1]['action_index']=source[0]['action_index']
    if fault=='role': source[1]['data_role']='validation'
    if fault=='layout': source[1]['layout_id']='other-layout'
    if fault=='validation':
        for row in source: row['data_role']='validation'
    with pytest.raises(ValueError): matched_schedule(source,updates=10,seed=0)


@pytest.mark.parametrize('updates,seed',[(0,1),(True,1),(1,-1),(1,True),(1,1.5)])
def test_invalid_schedule_parameters_rejected(updates,seed):
    with pytest.raises(ValueError): matched_schedule(rows(),updates=updates,seed=seed)


def test_non_development_dataset_role_rejected_before_read():
    with pytest.raises(ValueError,match='role'): AugmentedCausalDataset('test')
