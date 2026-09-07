"""Read native robot collision identity for evaluator-only startup/terminal checks."""
import numpy as np


def _array(value):
    if hasattr(value, 'detach'): value = value.detach().cpu().numpy()
    return np.asarray(value)


def capture_native_robot_geometry(robot):
    rows = []
    for geom in robot.geoms:
        position = _array(geom.get_pos()).reshape(-1, 3)
        quaternion = _array(geom.get_quat()).reshape(-1, 4)
        if position.shape != (1, 3) or quaternion.shape != (1, 4):
            raise ValueError('one actual native environment required')
        row = dict(geom_id=int(geom.idx), link_id=int(geom.link.idx), link_name=str(geom.link.name),
            geom_type=geom.type.name, data=_array(geom.data).tolist(),
            position_world_m=position[0].tolist(), quaternion_world_wxyz=quaternion[0].tolist(),
            friction=float(geom.friction), solver_parameters=_array(geom.sol_params).reshape(-1).tolist())
        if not all(np.isfinite(row[k]).all() for k in ('data', 'position_world_m', 'quaternion_world_wxyz', 'friction', 'solver_parameters')):
            raise ValueError('finite actual native geometry/material/solver data required')
        rows.append(row)
    return rows
