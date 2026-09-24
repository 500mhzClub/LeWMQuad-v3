"""Validate live sensor contracts before their tuple identities become JSON lists."""
import json
from lewm.dual_camera_visual_motion_development import current_dual_camera_pose
from lewm.measured_floor_transport_development import current_measured_floor_pose


def validate_live(live,recorded,comparison,policy,image,auxiliary,*,now_ns):
    if json.loads(json.dumps(live,allow_nan=False)) != recorded:
        raise ValueError('recorded decision must be the exact serialized live decision')
    if comparison['controller_recovered']:
        current_dual_camera_pose(live['original_visual_evidence'],policy,image,auxiliary,
            identity=(0,0,0),now_ns=now_ns)
        current_measured_floor_pose(live['evidence'],identity=(0,0,0),now_ns=now_ns)
