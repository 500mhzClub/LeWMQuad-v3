import math

import pytest

from lewm.memory.observed_exploration_development import ObservedExploration,PlaceFix,ExitObservation

SECOND=1_000_000_000


def fix(place,t,stable=True): return PlaceFix(f'fix-{place}-{t}',t*SECOND,place,stable)


def exit_at(memory,place,exit_id,t,bearing=0.):
    event=ExitObservation(f'exit-{place}-{exit_id}-{t}',t*SECOND,place,exit_id,bearing)
    memory.observe_exit(event); return event


def initial():
    memory=ObservedExploration(); memory.observe_place(fix('A',0)); return memory


def traverse(memory,source,target,exit_id,t):
    exit_at(memory,source,exit_id,t)
    action=f'action-{source}-{target}-{t}'; memory.begin(action,exit_id,now_ns=t*SECOND)
    memory.finish(action,fix(target,t+1),reached=True,viable_arrival=True)


def test_observation_and_selection_do_not_create_an_executed_edge():
    m=initial(); exit_at(m,'A','forward',0)
    assert m.decide(now_ns=0)['kind']=='TRAVERSE_EXIT'
    assert m.graph.places=={'A'} and m.graph.edge_summary('A','B')['attempts']==0
    m.begin('attempt','forward',now_ns=0)
    assert m.decide(now_ns=0)=={'kind':'EXECUTING'}
    assert not m.attempts and m.graph.places=={'A'}


def test_discover_beacon_then_physically_establish_directed_return():
    m=initial(); traverse(m,'A','B','out',0)
    m.observe_beacon('beacon','beacon-observation',timestamp_ns=SECOND)
    assert m.graph.route('A','B')==['A','B'] and m.graph.route('B','A') is None
    assert m.decide(now_ns=SECOND)['kind']=='NO_VERIFIED_RETURN'
    exit_at(m,'B','untried',1,math.pi)
    choice=m.decide(now_ns=SECOND)
    assert choice['kind']=='TRAVERSE_EXIT' and choice['reason']=='seek_return_route'
    m.begin('back','untried',now_ns=SECOND)
    m.finish('back',fix('A',2),reached=True,viable_arrival=True)
    assert m.graph.route('B','A')==['B','A']
    assert m.decide(now_ns=2*SECOND)=={'kind':'MISSION_COMPLETE','place_id':'A','observed_beacon_count':1}


def test_ambiguous_arrival_is_retained_and_later_fix_does_not_invent_edge():
    m=initial(); exit_at(m,'A','out',0); m.begin('attempt','out',now_ns=0)
    m.finish('attempt',fix(None,1),reached=True,viable_arrival=True)
    assert m.decide(now_ns=SECOND)=={'kind':'LOCALIZE'}
    assert m.exits[('A','out')]['status']=='UNCERTAIN' and not m.attempts[0]['qualified']
    m.observe_place(fix('B',2))
    assert m.graph.route('A','B') is None and m.graph.route('B','A') is None


@pytest.mark.parametrize('failed',['reached','viable_arrival','stable','same_place'])
def test_unqualified_arrival_never_qualifies_route(failed):
    m=initial(); exit_at(m,'A','out',0); m.begin('attempt','out',now_ns=0)
    target='A' if failed=='same_place' else 'B'
    m.finish('attempt',fix(target,1,stable=failed!='stable'),reached=failed!='reached',viable_arrival=failed!='viable_arrival')
    assert m.graph.edge_summary('A','B')['qualified_traversals']==0
    assert len(m.attempts)==1 and not m.attempts[0]['qualified']


def test_remote_frontier_uses_only_verified_routes_and_refreshes_old_bearing():
    m=initial(); exit_at(m,'A','other',0); traverse(m,'A','B','out',0)
    traverse(m,'B','A','back',1); traverse(m,'A','B','out',2)
    choice=m.decide(now_ns=3*SECOND)
    assert choice['kind']=='OBSERVE_EXIT' and choice['route']==['B','A'] and choice['exit_id']=='back'
    exit_at(m,'B','back',3,.4)
    choice=m.decide(now_ns=3*SECOND)
    assert choice['kind']=='TRAVERSE_EXIT' and choice['bearing_body_rad']==.4
    assert len(m.attempts)==3


def test_local_frontier_precedes_remote_observed_frontier():
    m=initial(); exit_at(m,'A','other',0); traverse(m,'A','B','out',0)
    exit_at(m,'B','new',1)
    assert m.decide(now_ns=SECOND)['exit_id']=='new'


def test_failed_latest_execution_disables_previous_route_without_automatic_retry():
    m=initial(); traverse(m,'A','B','out',0); traverse(m,'B','A','back',1)
    exit_at(m,'A','out',2); m.begin('failure','out',now_ns=2*SECOND)
    m.finish('failure',fix('A',3),reached=False,viable_arrival=True)
    assert m.graph.route('A','B') is None
    assert m.graph.edge_summary('A','B')['attempts']==2 and len(m.attempts)==3
    exit_at(m,'A','out',3)
    assert m.exits[('A','out')]['status']=='FAILED'
    with pytest.raises(ValueError,match='separate new qualification'): m.begin('retry','out',now_ns=3*SECOND)


def test_conflicting_exit_destination_quarantines_old_and_new_edges():
    m=initial(); traverse(m,'A','B','out',0); traverse(m,'B','A','back',1)
    exit_at(m,'A','out',2); m.begin('conflict','out',now_ns=2*SECOND)
    m.finish('conflict',fix('C',3),reached=True,viable_arrival=True)
    assert m.graph.route('A','B') is None and m.graph.route('A','C') is None
    assert m.exits[('A','out')]['status']=='CONFLICT' and m.attempts[-1]['association_conflict']
    assert m.fix.place_id=='C'  # keep the actual reported observation, not intended B


def test_stale_or_unstable_fix_cannot_dispatch_and_fresh_fix_requires_fresh_exit():
    m=initial(); exit_at(m,'A','out',0)
    assert m.decide(now_ns=SECOND)=={'kind':'OBSERVE_PLACE'}
    with pytest.raises(ValueError): m.begin('stale','out',now_ns=SECOND)
    m.observe_place(fix('A',1,stable=False))
    assert m.decide(now_ns=SECOND)=={'kind':'STABILIZE'}
    with pytest.raises(ValueError): m.begin('unstable','out',now_ns=SECOND)
    m.observe_place(fix('A',2))
    assert m.decide(now_ns=2*SECOND)['kind']=='OBSERVE_EXIT'
    with pytest.raises(ValueError,match='fresh exit'): m.begin('old-bearing','out',now_ns=2*SECOND)


def test_duplicate_beacon_observations_do_not_complete_multiple_beacon_task():
    m=ObservedExploration(required_beacons=2); m.observe_place(fix('A',0))
    m.observe_beacon('one','b1',timestamp_ns=0); m.observe_beacon('one','b1',timestamp_ns=0)
    m.observe_beacon('one','b2',timestamp_ns=0)
    assert len(m.beacons)==1 and len(m.beacons['one'])==2
    assert m.decide(now_ns=0)['kind']!='MISSION_COMPLETE'
    m.observe_beacon('two','b3',timestamp_ns=0)
    assert m.decide(now_ns=0)['kind']=='MISSION_COMPLETE'


def test_no_graph_or_frontiers_means_unexplored_not_mission_success():
    m=ObservedExploration(); assert m.decide(now_ns=0)=={'kind':'LOCALIZE'}
    m.observe_place(fix('A',0))
    assert m.decide(now_ns=0)['kind']=='NO_REACHABLE_OBSERVED_FRONTIER'
    assert m.decide(now_ns=0,return_home=True)['kind']=='HOME'


def test_unobserved_exit_wrong_place_or_teleport_during_traversal_rejected():
    m=initial()
    with pytest.raises(ValueError,match='unobserved exit'): m.begin('a','oracle-exit',now_ns=0)
    with pytest.raises(ValueError): exit_at(m,'B','out',0)
    with pytest.raises(ValueError): exit_at(m,'A','out',1)
    exit_at(m,'A','out',0); m.begin('a','out',now_ns=0)
    with pytest.raises(ValueError): m.observe_place(fix('B',1))
    with pytest.raises(ValueError): m.observe_beacon('b','bo',timestamp_ns=0)


def test_rejected_events_are_transactional_and_conflicting_ids_fail():
    m=initial(); before=m.snapshot()
    with pytest.raises(ValueError): m.observe_place(PlaceFix('fix-A-0',0,'B',True))
    assert m.snapshot()==before
    exit_at(m,'A','out',0); m.begin('a','out',now_ns=0); before=m.snapshot()
    with pytest.raises(ValueError): m.finish('a',fix('B',0),reached=True,viable_arrival=True)
    with pytest.raises(ValueError): m.finish('different',fix('B',1),reached=True,viable_arrival=True)
    assert m.snapshot()==before
    m.finish('a',fix('B',1),reached=True,viable_arrival=True)
    with pytest.raises(ValueError): m.decide(now_ns=0)
    with pytest.raises(ValueError): m.observe_place(PlaceFix('new-stale',0,'A',True))


@pytest.mark.parametrize('bad',[True,-1,float('nan'),1.5])
def test_bad_clock_rejected(bad):
    with pytest.raises(ValueError): PlaceFix('fix',bad,'A',True)


@pytest.mark.parametrize('bad',[True,float('nan'),float('inf'),4.])
def test_bad_observed_bearing_rejected(bad):
    with pytest.raises(ValueError): ExitObservation('exit',0,'A','out',bad)


def test_snapshot_mutation_does_not_rewrite_runtime_memory():
    m=initial(); traverse(m,'A','B','out',0); snapshot=m.snapshot()
    snapshot['attempts'][0]['qualified']=False; snapshot['exits'][0]['observation']['bearing_body_rad']=99.
    assert m.attempts[0]['qualified'] and m.exits[('A','out')]['observation'].bearing_body_rad==0.
