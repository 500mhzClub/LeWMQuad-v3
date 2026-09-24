from types import SimpleNamespace
import numpy as np
from lewm.initial_panorama_development import InitialPanorama


def test_initial_survey_requires_every_heading_and_fresh_map_before_completion():
    survey=InitialPanorama();snapshot=SimpleNamespace(measured_ns=0,frame=0)
    first=survey.advance(snapshot,np.zeros(2),0.,100)
    assert not first['complete'] and first['completed_view_stages']==[]
    for i in range(9):
        heading=survey.state['heading_rad'];now=200+i*200
        # Neither a stale map nor the wrong heading completes a new stage.
        before=len(survey.state['completed_view_stages'])
        survey.advance(snapshot,np.zeros(2),heading+.3,now)
        assert len(survey.state['completed_view_stages'])==before
        survey.advance(snapshot,np.zeros(2),heading,now)
        assert not survey.complete
        snapshot.measured_ns=now;snapshot.frame+=1
        result=survey.advance(snapshot,np.zeros(2),heading,now+100)
        assert result['complete']==(i==8)
    assert len(result['completed_view_stages'])==9
    assert first['completed_view_stages']==[]
