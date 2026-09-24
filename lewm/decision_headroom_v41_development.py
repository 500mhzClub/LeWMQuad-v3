"""V4.1 evaluator amendments. No controller, candidate or model changes."""
import heapq
import math
import numpy as np
from lewm.physical_execution_development import rotation_xyzw
from lewm.decision_headroom_reference_development import ReferenceGeometry, wrap


class SecondaryReference:
    """Escape inflation through point-free space, then use the unchanged field.

    Nearest means shortest path length on the existing 20-mm eight-neighbour
    grid, with lexicographic ties. No search over target cost or action outcomes.
    The point-free leg is a reference convention, not robot clearance evidence.
    """
    def __init__(self, primary):
        self.primary = primary
        self.point = ReferenceGeometry(primary.walls, primary.bounds, primary.target,
            radius_m=1e-12, clearance_m=0., resolution_m=primary.resolution)
        # Exactly zero inflation, without relaxing ReferenceGeometry's public
        # positive-radius constructor for any existing caller.
        self.point.radius = 0.
        self.free = None

    def distance_and_heading(self, point):
        p = self.primary
        original = p.distance_and_heading(point)
        if original['valid'] or float(p.footprint_clearance(point)) >= 0:
            return original
        point = np.asarray(point, float)
        if float(self.point.footprint_clearance(point)) <= 0:
            return dict(valid=False, reason='SECONDARY_ENDPOINT_NOT_POINT_FREE')
        if p.free is None:
            p._build_field()
        if self.free is None:
            grid = np.indices(p.shape).transpose(1, 2, 0)*p.resolution+p.bounds[0]
            self.free = self.point.footprint_clearance(grid) >= p.resolution/math.sqrt(2)
        distances = {}; queue = []
        for i in p._nearby(point):
            if self.free[i] and self.point.segment_clear(point, p._point(i)):
                length = float(np.linalg.norm(p._point(i)-point))
                distances[i] = length
                heapq.heappush(queue, (length, *i))
        while queue:
            length, x, y = heapq.heappop(queue)
            if distances[x,y] != length:
                continue
            if p.free[x,y]:
                entry = p._point((x,y))
                tail = p.distance_and_heading(entry)
                if not tail['valid']:
                    # Do not choose a more favourable/different entry on failure.
                    return dict(valid=False, reason='NEAREST_INFLATION_FREE_CELL_HAS_NO_TARGET_PATH',
                        entry_xy=entry.tolist(), entry_path_m=length)
                return dict(tail, distance_m=tail['distance_m']+length,
                    secondary_inflation_escape=True, entry_xy=entry.tolist(),
                    entry_path_m=length, heading_convention='unchanged primary heading at entry cell',
                    nearest_rule='shortest non-inflated grid path; lexicographic ties')
            for dx in (-1,0,1):
                for dy in (-1,0,1):
                    if not (dx or dy): continue
                    xx, yy = x+dx, y+dy
                    if not (0 <= xx < p.shape[0] and 0 <= yy < p.shape[1]) or not self.free[xx,yy]: continue
                    if dx and dy and not (self.free[x+dx,y] and self.free[x,y+dy]): continue
                    new = length+p.resolution*math.hypot(dx,dy)
                    if new < distances.get((xx,yy), math.inf):
                        distances[xx,yy] = new
                        heapq.heappush(queue, (new,xx,yy))
        return dict(valid=False, reason='NO_REACHABLE_INFLATION_FREE_CELL')


def localisation(packet, initial_pose, true_pose):
    """Evaluator-only fixed initial anchor; never correct a controller input."""
    try:
        B = np.asarray(packet['observed_map'].map_from_initial, float)
        q = np.asarray(packet['observed_position'], float)
        Q = np.asarray(packet['observed_rotation'], float)
        initial_pose, true_pose = np.asarray(initial_pose), np.asarray(true_pose)
        R = rotation_xyzw(initial_pose[3:])
        believed = initial_pose[:3] + R @ B.T @ q
        rotation = R @ B.T @ Q
        actual = rotation_xyzw(true_pose[3:])
        translation = believed-true_pose[:3]
        yaw = wrap(math.atan2(rotation[1,0],rotation[0,0])-math.atan2(actual[1,0],actual[0,0]))
        angle = math.acos(float(np.clip((np.trace(rotation.T@actual)-1)/2,-1,1)))
        norm = float(np.linalg.norm(translation[:2]))
        if not np.isfinite(np.r_[translation,yaw,angle]).all(): raise ValueError('nonfinite pose')
        return dict(status='available', believed_position_world_m=believed.tolist(),
            true_position_world_m=true_pose[:3].tolist(), translation_error_world_m=translation.tolist(),
            horizontal_error_m=norm, translation_error_m=float(np.linalg.norm(translation)),
            yaw_error_rad=yaw, rotation_error_rad=angle,
            position_stratum='le_20mm' if norm<=.02 else '20_to_100mm' if norm<=.10 else 'gt_100mm',
            yaw_stratum='le_5deg' if abs(yaw)<=math.radians(5) else '5_to_15deg' if abs(yaw)<=math.radians(15) else 'gt_15deg',
            evaluator_only=True, anchor='unchanged first-source-camera physical pose')
    except (ValueError, KeyError, AttributeError) as exc:
        return dict(status='unresolved', reason=repr(exc), position_stratum='unresolved', yaw_stratum='unresolved')


def paired_filter(state_rows, motion, quantity):
    """Paired numerator difference on one state/candidate mask and denominator."""
    numerator = denominator = 0.; matched_states = matched_candidates = 0
    for w, r in state_rows:
        a = r['filter_audit'].get(motion, {}).get('operating')
        b = r['filter_audit'].get('R2', {}).get('operating')
        if not a or not b: continue
        if quantity == 'excluded_safe':
            supported = False
            for aa, bb in zip(a['candidates'],b['candidates'],strict=True):
                if aa['action'] != bb['action']: raise ValueError('candidate identity mismatch')
                if aa['resolved'] and bb['resolved'] and aa['safety']==bb['safety']=='safe':
                    numerator += w*(int(aa['excluded_but_safe'])-int(bb['excluded_but_safe']))
                    denominator += w; matched_candidates += 1; supported = True
            matched_states += supported
        else:
            aa,bb = a['all_movement_excluded_despite_safe'],b['all_movement_excluded_despite_safe']
            if aa is not None and bb is not None:
                numerator += w*(int(aa)-int(bb)); denominator += w; matched_states += 1
    return dict(value=numerator/denominator if denominator else None,
        numerator=numerator, denominator=denominator, matched_states=matched_states,
        matched_safe_candidates=matched_candidates, common_mask=True)


def binding_event(value):
    """B(i): event restricted to motion-gate binding, with original denominator.

    All known-safe movements must have a motion-dependent binding exclusion;
    observation-only binding never counts as motion-gate complete exclusion.
    Missing candidate safety prevents attribution of a positive event.
    """
    event = value['all_movement_excluded_despite_safe']
    if event is not True: return event
    if any(not c['resolved'] for c in value['candidates']): return None
    safe = [c for c in value['candidates'] if c['safety']=='safe']
    return bool(safe) and all(c['binding_rule'] in ('MOTION_MEMORY_CLEARANCE','MOTION_STOPPING_PROJECTION') for c in safe)
