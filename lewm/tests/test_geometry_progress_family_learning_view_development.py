"""Matched schedules do not overweight long recordings or touch transfer roles."""
from collections import Counter
from copy import deepcopy
import pytest
from lewm.geometry_progress_layout_family_development import assignments
from lewm.geometry_progress_family_causal_windows_development import OFFSETS_TICKS
from lewm.geometry_progress_family_learning_view_development import FamilyWindowView


def windows():
    rows=[]
    for trial,c in assignments().items():
        for offset in OFFSETS_TICKS:
            available=offset==0 or c['action']=='hold'
            rows.append(dict(trial=trial,**c,offset_ticks=offset,window_id=f'{trial}/offset_{offset:02d}',
                available=available,reason=None if available else 'MISSING_ACTUAL_CONTEXT',targets=[{}]*8 if available else None))
    return rows


def test_long_safe_episodes_do_not_receive_more_training_draws():
    view=FamilyWindowView(windows());schedule=view.schedule(updates=1200,batch_size=6,seed=2026090950)
    draws=[view.windows[i] for b in schedule['batches'] for i in b]
    counts=Counter(w['trial'] for w in draws)
    assert len(counts)==48 and set(counts.values())=={150}
    assert Counter(w['action'] for w in draws)=={a:1200 for a in {w['action'] for w in draws}}
    assert all(w['data_role']=='train' for w in draws)
    assert schedule==view.schedule(updates=1200,batch_size=6,seed=2026090950)


def test_schedule_is_invariant_to_native_outcome_values():
    rows=windows();a=FamilyWindowView(rows).schedule(updates=8,batch_size=6,seed=7)
    for w in rows:
        if w['available']:w['targets']=[dict(motion=[999.,999.,999.],contact=1.)]*8
    assert FamilyWindowView(rows).schedule(updates=8,batch_size=6,seed=7)==a


def test_absent_episode_cannot_be_silently_dropped():
    rows=windows();trial=next(t for t,c in assignments().items() if c['data_role']=='train')
    for w in rows:
        if w['trial']==trial:w.update(available=False,reason='MISSING_ACTUAL_CONTEXT',targets=None)
    with pytest.raises(ValueError,match='disappear'):FamilyWindowView(rows).schedule(updates=8,batch_size=6,seed=7)


def test_input_snapshot_is_private_and_initial_transfer_population_is_complete():
    rows=windows();view=FamilyWindowView(rows);rows[0]['available']=not rows[0]['available']
    assert view.windows[0]['available']!=rows[0]['available']
    assert len(view.indices('geometry_transfer',initial_only=True))==48
    with pytest.raises(ValueError):view.indices('sealed')
