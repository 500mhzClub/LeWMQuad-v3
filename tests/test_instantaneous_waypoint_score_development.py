from copy import deepcopy
import math
import numpy as np
from lewm.instantaneous_waypoint_score_development import instantaneous_scores,replace_ranking
from lewm.waypoint_alignment_planning_development import score_waypoint_alignment


def potential(goal,xy,yaw):
    d=np.asarray(goal)-xy
    e=math.atan2(d[1],d[0])-yaw
    return np.linalg.norm(d)+min(.35,np.linalg.norm(goal))*abs(math.atan2(math.sin(e),math.cos(e)))


def test_scores_match_directional_derivative_of_existing_objective():
    from lewm.geometry_progress_pilot_development import candidate_commands
    for goal in ([1.,.3],[.2,-.1],[1.,0.],[-1.,0.]):
        for row in instantaneous_scores(goal):
            v,_,w=candidate_commands(row['action'])[0];dt=1e-7
            measured=(potential(goal,np.zeros(2),0.)-potential(goal,np.array([v*dt,0.]),w*dt))/dt
            assert abs(measured-row['utility_m']/.4)<2e-7


def test_view_alignment_and_terminal_command_duration():
    positive=instantaneous_scores([0.,0.],scan_error=.6)
    assert max((r for r in positive if r['eligible_for_view']),key=lambda r:r['utility_m'])['action']=='left_turn'
    aligned=instantaneous_scores([0.,0.],scan_error=0.)
    assert max((r for r in aligned if r['eligible_for_view']),key=lambda r:r['utility_m'])['action']=='hold'
    full=instantaneous_scores([.08,.02]);short=instantaneous_scores([.08,.02],pulse=True)
    assert short[1]['utility_m']==full[1]['utility_m']/4
    assert short[4]['utility_m']==full[4]['utility_m']


def test_rank_utilities_ignore_predictions_and_preserve_forecast_gate_evidence():
    p=np.zeros((6,8,5));p[:,:,3]=1.;p[:,:,4]=-1000.
    a=score_waypoint_alignment(p,[1.,.2],delay_ticks=3,commit_ticks=4)
    p[1,:,0]=np.linspace(0.,-.3,8)
    b=score_waypoint_alignment(p,[1.,.2],delay_ticks=3,commit_ticks=4)
    saved=deepcopy(b)
    x,y=(replace_ranking(v,pulse=False) for v in (a,b))
    assert b==saved
    assert x['instantaneous_ranking']['rows']==y['instantaneous_ranking']['rows']
    assert [r['utility_m'] for r in x['candidates']]==[r['utility_m'] for r in y['candidates']]
    assert x['action']==y['action']
    assert x['candidates'][1]['predicted_progress_during_commit_m']!=y['candidates'][1]['predicted_progress_during_commit_m']
    assert y['instantaneous_ranking']['forecast_candidates']==saved['candidates']
