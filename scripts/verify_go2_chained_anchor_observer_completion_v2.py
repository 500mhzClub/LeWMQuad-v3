"""V1 completion check with an explicit JSON-to-runtime identity adapter only."""
from pathlib import Path
from types import FunctionType, SimpleNamespace

from lewm.causal_sensor_state import SensorContractError, _identity
from scripts import verify_go2_chained_anchor_observer_completion_v1 as previous

SOURCE = 'scripts/verify_go2_chained_anchor_observer_completion_v2.py'
TEST = 'lewm/tests/test_chained_anchor_observer_completion_v2_development.py'
OUTPUT = Path('docs/go2_chained_anchor_observer_completion_v2_2026-09-11.json')
RESULT_SHA, LAUNCH_SHA, OWNER = previous.RESULT_SHA, previous.LAUNCH_SHA, previous.OWNER
run = previous.run


def check_serialized_pose(evidence, *args, **kwargs):
    if type(evidence.get('identity')) is not list:
        raise SensorContractError('serialized episode identity must be a JSON array')
    identity = _identity(tuple(evidence['identity']))
    return run.current_dual_camera_pose(dict(evidence, identity=identity), *args, **kwargs)


def main():
    # Private function globals retain every original comparison, raw packet,
    # artifact, source and completed-report check. Imported modules are unchanged.
    view = SimpleNamespace(**vars(run))
    view.current_dual_camera_pose = check_serialized_pose
    worker = FunctionType(previous.main.__code__, previous.main.__globals__ | dict(
        run=view, SOURCE=SOURCE, TEST=TEST, OUTPUT=OUTPUT), 'verify_serialized_observer_completion')
    return worker()


if __name__ == '__main__':
    main()
