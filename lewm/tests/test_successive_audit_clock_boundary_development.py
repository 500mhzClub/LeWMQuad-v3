import pytest

from lewm.successive_audit_clock_boundary_development import expected_ingested


def fixture():
    return ([{'physical_sample_index':99+i*50,'timestamp_s':.2+i*.1} for i in range(3)],
        [{'pre_sample_index':99,'post_sample_index':149,'stage':'control'},
         {'pre_sample_index':149,'post_sample_index':199,'stage':'control'}])


def test_complete_nonterminal_last_command_is_ingested():
    camera,tape=fixture()
    assert expected_ingested(camera,tape,prefix_end_time=.2,terminal_index=449,stop_reason=None)==[0,1,2]


@pytest.mark.parametrize('reason',['DISALLOWED_CONTACT','BODY_STABILITY_LIMIT'])
def test_native_terminal_on_fiftieth_sample_is_not_post_ingested(reason):
    camera,tape=fixture()
    assert expected_ingested(camera,tape,prefix_end_time=.2,terminal_index=199,stop_reason=reason)==[0,1]


def test_early_native_terminal_and_release_contact_do_not_skip_valid_history():
    camera,tape=fixture(); tape[-1]['post_sample_index']=198; camera[-1]['physical_sample_index']=198
    assert expected_ingested(camera,tape,prefix_end_time=.2,terminal_index=198,stop_reason='DISALLOWED_CONTACT')==[0,1]
    camera,tape=fixture()
    assert expected_ingested(camera,tape,prefix_end_time=.2,terminal_index=449,stop_reason='DISALLOWED_CONTACT')==[0,1,2]


def test_nonterminal_missing_packet_still_fails():
    camera,tape=fixture()
    with pytest.raises(ValueError,match='missing completed'):
        expected_ingested(camera[:1]+camera[2:],tape,prefix_end_time=.2,terminal_index=449,stop_reason=None)
