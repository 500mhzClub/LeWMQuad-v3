"""Regression for the actual recorded execution schema and its frozen launch."""
from copy import deepcopy
import json
import pytest
from scripts import verify_go2_sustained_hold_reorientation_raw_completion_v2 as check


@pytest.mark.parametrize('fault', [None, 'count', 'owner', 'launch_hash', 'closure'])
def test_actual_receipt_gets_sources_only_from_hash_verified_original_launch(monkeypatch, fault):
    execution = json.loads((check.ROOT/check.original.EXECUTION).read_text())
    assert 'source_sha256' not in execution
    table = {str(i):'sha' for i in range(execution['source_count'])}
    launch = dict(source_sha256=table, owner_pid=execution['owner']['pid'], boot_id=execution['boot_id'])
    sources = dict(table); before = deepcopy(execution); verified = []
    monkeypatch.setattr(check.original, 'require_ended', lambda:None)
    monkeypatch.setattr(check, 'verify_artifacts', lambda root, ids:verified.append(ids))
    monkeypatch.setattr(check, 'read_json', lambda *a:launch)
    if fault == 'count': execution['source_count'] += 1
    if fault == 'owner': launch['owner_pid'] += 1
    if fault == 'launch_hash': execution['launch_sha256'] = 'changed'
    if fault == 'closure': sources['0'] = 'changed'
    if fault:
        with pytest.raises(ValueError): check.execution_with_sources(execution, sources)
    else:
        result = check.execution_with_sources(execution, sources)
        assert execution == before and result['source_sha256'] == table
        assert result['source_sha256'] is not table
        assert verified == [{'launch.json':execution['launch_sha256']}]
