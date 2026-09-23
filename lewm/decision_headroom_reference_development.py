"""Evaluator-only physical reference; no model, observed-map or scorer imports.

The reference is relative to declared disk inflation and a discrete geodesic
grid. It is not an unrestricted optimal cost-to-go. Phase 1 qualifies its
physical-unit components on prewritten sanity cases before protocol freeze.
"""
import heapq
import math

import numpy as np


def wrap(angle):
    return math.atan2(math.sin(angle), math.cos(angle))


def fixed_target_world(packet, initial_physical_pose):
    """Map the recorded target through the source's single initial-frame anchor.

    The initial pose is evaluator-only physics at the first source RGB frame.
    Never re-anchor a target at the current true pose: doing so would silently
    correct source perception/routing errors during the audit.
    """
    from lewm.physical_execution_development import rotation_xyzw

    target = packet['active_target']
    if target['kind'] == 'NO_XY_ROUTE_TARGET':
        return dict(valid=False, reason='NO_XY_ROUTE_TARGET', cost_s=None)
    xy = np.asarray(target['xy'], float)
    initial = np.asarray(initial_physical_pose, float)
    if xy.shape != (2,) or initial.shape != (7,) or not np.isfinite(xy).all() or not np.isfinite(initial).all():
        raise ValueError('finite recorded target and initial physical pose required')
    if target['kind'] == 'initial_frame_xy':
        point_initial = np.r_[xy, 0.]
    elif target['kind'] == 'observed_map_xy':
        B = np.asarray(packet['observed_map'].map_from_initial, float)
        q = np.asarray(packet['observed_position'], float)
        if B.shape != (3,3) or q.shape != (3,):
            raise ValueError('recorded map-to-initial rotation and position required')
        # The source route target uses zero map-frame vertical displacement
        # from q. Preserve that plane at this decision, not a true-map snap.
        point_initial = B.T @ np.r_[xy, q[2]]
    else:
        raise ValueError('unknown recorded target coordinate convention')
    world = initial[:3] + rotation_xyzw(initial[3:]) @ point_initial
    return dict(valid=True, target_xy_world=world[:2].tolist(),
        source_target_kind=target['kind'], initial_frame_anchor_only=True,
        current_true_pose_reanchoring=False, true_geometry_snapping=False)


class ReferenceGeometry:
    def __init__(self, walls, bounds, target_xy, *, radius_m, clearance_m, resolution_m):
        self.walls = walls
        self.bounds = np.asarray(bounds, float)
        self.target = np.asarray(target_xy, float)
        self.radius = float(radius_m)
        self.clearance = float(clearance_m)
        self.resolution = float(resolution_m)
        if self.bounds.shape != (2, 2) or np.any(self.bounds[1] <= self.bounds[0]):
            raise ValueError('finite rectangular world bounds required')
        if not np.isfinite(self.bounds).all() or not np.isfinite(self.target).all():
            raise ValueError('finite true geometry required')
        if min(self.radius, self.resolution) <= 0 or self.clearance < 0:
            raise ValueError('positive footprint/grid scale and nonnegative clearance required')
        self.shape = tuple((np.floor((self.bounds[1] - self.bounds[0]) / self.resolution).astype(int) + 1).tolist())
        self.distance_field = None
        self.free = None

    def footprint_clearance(self, xy):
        points = np.asarray(xy, float)
        flat = points.reshape(-1, 2)
        margin = np.minimum(flat - self.bounds[0], self.bounds[1] - flat).min(axis=1)
        for wall in self.walls:
            c, s = math.cos(wall['yaw']), math.sin(wall['yaw'])
            local = (flat - wall['center']) @ np.array([[c, -s], [s, c]])
            delta = np.abs(local) - np.asarray(wall['size']) / 2
            distance = np.linalg.norm(np.maximum(delta, 0), axis=1) + np.minimum(np.max(delta, axis=1), 0)
            margin = np.minimum(margin, distance)
        return (margin - self.radius).reshape(points.shape[:-1])

    def segment_clear(self, start, end):
        start, end = np.asarray(start), np.asarray(end)
        distance = np.linalg.norm(end - start)
        count = max(1, int(math.ceil(distance / (self.resolution / 4))))
        samples = start + np.linspace(0, 1, count + 1)[:, None] * (end - start)
        # Distance to a fixed obstacle is 1-Lipschitz. This is a conservative
        # bound along the interpolated straight segment, not just its vertices.
        lower = float(np.min(self.footprint_clearance(samples))) - distance / (2 * count)
        return lower >= self.clearance

    def _point(self, index):
        return self.bounds[0] + np.asarray(index) * self.resolution

    def _nearby(self, point):
        middle = np.rint((point - self.bounds[0]) / self.resolution).astype(int)
        for dx in range(-1, 2):
            for dy in range(-1, 2):
                index = (int(middle[0] + dx), int(middle[1] + dy))
                if all(0 <= index[k] < self.shape[k] for k in (0, 1)):
                    yield index

    def _build_field(self):
        grid = np.indices(self.shape).transpose(1, 2, 0) * self.resolution + self.bounds[0]
        # Half a diagonal cell protects all interpolation within a free cell.
        self.free = self.footprint_clearance(grid) >= self.clearance + self.resolution / math.sqrt(2)
        distances = np.full(self.shape, np.inf)
        queue = []
        for index in self._nearby(self.target):
            if self.free[index] and self.segment_clear(self.target, self._point(index)):
                d = float(np.linalg.norm(self._point(index) - self.target))
                distances[index] = d
                heapq.heappush(queue, (d, *index))
        directions = [(dx, dy, self.resolution * math.hypot(dx, dy))
                      for dx in (-1, 0, 1) for dy in (-1, 0, 1) if dx or dy]
        while queue:
            d, x, y = heapq.heappop(queue)
            if d != distances[x, y]:
                continue
            for dx, dy, length in directions:
                xx, yy = x + dx, y + dy
                if not (0 <= xx < self.shape[0] and 0 <= yy < self.shape[1]) or not self.free[xx, yy]:
                    continue
                if dx and dy and not (self.free[x + dx, y] and self.free[x, y + dy]):
                    continue
                new = d + length
                if new < distances[xx, yy]:
                    distances[xx, yy] = new
                    heapq.heappush(queue, (new, xx, yy))
        self.distance_field = distances

    def distance_and_heading(self, point):
        point = np.asarray(point, float)
        if self.footprint_clearance(self.target) < self.clearance:
            return dict(valid=False, reason='TARGET_PHYSICALLY_INVALID')
        if self.footprint_clearance(point) < self.clearance:
            return dict(valid=False, reason='ENDPOINT_OUTSIDE_REFERENCE_FREE_SPACE')
        vector = self.target - point
        direct = float(np.linalg.norm(vector))
        if direct <= 1e-12:
            return dict(valid=True, distance_m=0., heading_rad=None, direct_segment=True)
        if self.segment_clear(point, self.target):
            return dict(valid=True, distance_m=direct, heading_rad=math.atan2(vector[1], vector[0]), direct_segment=True)
        if self.distance_field is None:
            self._build_field()
        options = []
        for index in self._nearby(point):
            d = self.distance_field[index]
            if np.isfinite(d) and self.segment_clear(point, self._point(index)):
                options.append((d + np.linalg.norm(self._point(index) - point), index))
        if not options:
            return dict(valid=False, reason='NO_REFERENCE_GEODESIC')
        distance, index = min(options)
        # A fixed 0.20-m descent lookahead avoids reporting a grid-edge bearing
        # from an arbitrarily tiny endpoint-to-grid offset as the next heading.
        current = index
        for _ in range(int(math.ceil(.20 / self.resolution))):
            neighbors = [j for j in self._nearby(self._point(current))
                         if self.distance_field[j] < self.distance_field[current]
                         and self.segment_clear(self._point(current), self._point(j))]
            if not neighbors:
                break
            candidate = min(neighbors, key=lambda j:(self.distance_field[j], j))
            if not self.segment_clear(point, self._point(candidate)):
                break
            current = candidate
        vector = self._point(current) - point
        if np.linalg.norm(vector) <= 1e-12:
            return dict(valid=False, reason='UNDEFINED_GEODESIC_DESCENT')
        return dict(valid=True, distance_m=float(distance), heading_rad=math.atan2(vector[1], vector[0]), direct_segment=False)


def reference_cost(geometry, trace, *, arrival_settling, parameters):
    """Score one physical branch; unacceptable branches have no scalar cost.

    trace contains source plus all actual samples through 800 ms. Every cost
    component is retained. No safety penalty is mixed into progress.
    """
    xy = np.asarray(trace['xy'], float)
    if xy.ndim != 2 or xy.shape[1] != 2 or not np.isfinite(xy).all():
        raise ValueError('finite physical XY trace required')
    offsets = np.asarray(trace['offset_ns'], np.int64)
    if offsets.shape != (len(xy),) or offsets[0] != 0 or np.any(np.diff(offsets) <= 0):
        raise ValueError('ordered physical trace relative to the decision timestamp required')
    for name in ('yaw', 'velocity_xy', 'yaw_rate', 'disallowed_contact'):
        values = np.asarray(trace[name])
        if len(values) != len(xy) or not np.isfinite(values).all():
            raise ValueError(f'complete finite physical samples required: {name}')
    clearance = float(np.min(geometry.footprint_clearance(xy)))
    max_step = float(np.max(np.linalg.norm(np.diff(xy, axis=0), axis=1))) if len(xy) > 1 else 0.
    lower = clearance - max_step / 2
    contact = bool(np.asarray(trace['disallowed_contact']).any())
    acceptable = not contact and lower >= geometry.clearance
    result = dict(acceptable=acceptable, disallowed_contact=contact,
        sampled_footprint_clearance_m=clearance, interpolated_sweep_clearance_lower_bound_m=lower,
        cost_s=None, components=None)
    if offsets[-1] != 800_000_000 or not all(h in offsets for h in range(100_000_000, 800_000_001, 100_000_000)):
        result.update(acceptable=False if contact else None, reason='INCOMPLETE_BRANCH_HORIZON')
        return result
    endpoint = geometry.distance_and_heading(xy[-1])
    result['reference_endpoint'] = endpoint
    if not acceptable or not endpoint['valid']:
        return result
    yaw = float(np.asarray(trace['yaw'])[-1])
    velocity = np.asarray(trace['velocity_xy'], float)[-1]
    omega = float(np.asarray(trace['yaw_rate'])[-1])
    settling = bool(arrival_settling and endpoint['distance_m'] <= parameters['arrival_radius_m'])
    heading = endpoint['heading_rad']
    error = 0. if heading is None or settling else wrap(heading - yaw)
    direction = np.array([math.cos(heading), math.sin(heading)]) if heading is not None else np.zeros(2)
    components = dict(remaining_travel_s=endpoint['distance_m'] / parameters['nominal_travel_speed_m_s'],
        heading_alignment_s=abs(error) / parameters['nominal_turn_speed_rad_s'],
        reverse_motion_braking_s=0. if settling else max(0., -float(velocity @ direction)) / parameters['linear_braking_acceleration_m_s2'],
        opposing_turn_braking_s=0. if settling else max(0., -omega * np.sign(error)) / parameters['angular_braking_acceleration_rad_s2'],
        arrival_linear_settling_s=max(0., float(np.linalg.norm(velocity)) - parameters['quiet_linear_speed_m_s']) / parameters['linear_braking_acceleration_m_s2'] if settling else 0.,
        arrival_angular_settling_s=max(0., abs(omega) - parameters['quiet_angular_speed_rad_s']) / parameters['angular_braking_acceleration_rad_s2'] if settling else 0.)
    result.update(cost_s=float(sum(components.values())), components=components, arrival_settling_applied=settling)
    return result
