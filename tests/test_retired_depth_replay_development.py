import json
import pytest
from scripts.in_memory_public_replay_development import PublicReplay


@pytest.mark.parametrize('status', ['DEPTH_RETIREMENT_IN_PROGRESS', 'DEPTH_RETIRED'])
def test_retired_recording_reports_retention_before_loading_missing_arrays(tmp_path, status):
    native = tmp_path/'native'; native.mkdir()
    (tmp_path/'depth_retention.json').write_text(json.dumps(dict(
        status=status, full_sensor_replay_available=False)))
    with pytest.raises(ValueError, match='Depth recording intentionally retired'):
        PublicReplay(native)
