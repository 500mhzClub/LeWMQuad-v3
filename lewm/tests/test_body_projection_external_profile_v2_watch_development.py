from types import FunctionType
import pytest
from scripts import await_go2_body_projection_external_profile_completion_v2 as watch
from lewm.tests import test_body_projection_external_profile_watch_development as original


@pytest.mark.parametrize('mode',['complete','profile_failure','checker_failure','already_checked'])
def test_v2_watch_preserves_once_only_completion_and_failure_paths(monkeypatch,tmp_path,mode):
    test=original.test_exactly_one_checker_after_ended_parent
    clone=FunctionType(test.__code__,test.__globals__ | {'watch':watch},test.__name__)
    clone(monkeypatch,tmp_path,mode)
