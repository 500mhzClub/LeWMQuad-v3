"""Training-only empirical pulse dynamics; inference has no target argument.

Exact action/time cells, no interpolation or evaluation-label fallback. Repeated
training rows retain their exposure weights. Missing cells stay explicitly
unavailable; callers must not silently exclude them from a model comparison.
"""
import math
import numpy as np


def validate_queries(actions, offsets, active):
    actions = np.asarray(actions)
    offsets = np.asarray(offsets)
    active = np.asarray(active)
    if (actions.ndim != 1 or actions.dtype.kind not in 'iu'
            or not np.isin(actions, range(6)).all()
            or offsets.shape != (len(actions), 8) or offsets.dtype.kind not in 'iu'
            or active.shape != offsets.shape or active.dtype != bool):
        raise ValueError('integer action/time arrays and boolean eight-slot mask required')
    if ((offsets[active] <= 0).any() or (offsets[~active] != 0).any()
            or (active[:, 1:] & ~active[:, :-1]).any()):
        raise ValueError('positive exact known times and zero unknown times required')
    for times in offsets:
        if (np.diff(times[times > 0]) <= 0).any():
            raise ValueError('strictly increasing known times required')
    return actions, offsets, active


class ActionTimeMean:
    def __init__(self, cells, training_draws):
        self.cells = cells
        self.training_draws = training_draws

    @classmethod
    def fit(cls, actions, offsets, active, targets, *, roles):
        actions, offsets, active = validate_queries(actions, offsets, active)
        if not len(actions) or len(roles) != len(actions) or any(r != 'train' for r in roles):
            raise ValueError('nonempty training-role draws only')
        motion = np.asarray(targets['motion'], dtype=float)
        contact = np.asarray(targets['contact'], dtype=float)
        mv = np.asarray(targets['motion_valid'])
        cv = np.asarray(targets['contact_valid'])
        if (motion.shape != (*active.shape, 3) or contact.shape != active.shape
                or mv.shape != active.shape or cv.shape != active.shape
                or mv.dtype != bool or cv.dtype != bool or ((mv | cv) & ~active).any()
                or (mv & ~cv).any() or not np.isfinite(motion[mv]).all()
                or not np.isin(contact[cv], [0, 1]).all() or (contact[mv] != 0).any()):
            raise ValueError('observed collision-free motion and independent contact masks required')
        cells = {}
        for action in sorted(set(actions.tolist())):
            for ns in sorted(set(offsets[active & (actions[:, None] == action)].tolist())):
                cell = active & (actions[:, None] == action) & (offsets == ns)
                m, c = motion[cell & mv], contact[cell & cv]
                value = None
                if len(m):
                    value = [*m[:, :2].mean(0).tolist(),
                             float(np.sin(m[:, 2]).mean()), float(np.cos(m[:, 2]).mean())]
                probability = float(c.mean()) if len(c) else None
                clipped = None if probability is None else np.clip(probability, 1e-6, 1-1e-6)
                cells[(action, ns)] = dict(action_index=action, offset_ns=ns,
                    training_motion_count=len(m), training_contact_count=len(c),
                    motion_mean=value, contact_frequency=probability,
                    contact_logit=None if clipped is None else math.log(clipped/(1-clipped)))
        return cls(cells, len(actions))

    def predict(self, actions, offsets, active):
        actions, offsets, active = validate_queries(actions, offsets, active)
        prediction = np.full((*active.shape, 5), np.nan)
        prediction[~active] = 0
        missing = []
        for i, h in zip(*np.nonzero(active), strict=True):
            key = (int(actions[i]), int(offsets[i, h]))
            cell = self.cells.get(key)
            motion_ok = cell is not None and cell['motion_mean'] is not None
            contact_ok = cell is not None and cell['contact_logit'] is not None
            if motion_ok: prediction[i, h, :4] = cell['motion_mean']
            if contact_ok: prediction[i, h, 4] = cell['contact_logit']
            if not (motion_ok and contact_ok):
                missing.append(dict(row=int(i), horizon=int(h), action_index=key[0],
                    offset_ns=key[1], motion_missing=not motion_ok, contact_missing=not contact_ok))
        return prediction, missing

    def record(self):
        return dict(training_draws=self.training_draws,
            cells=[dict(self.cells[k]) for k in sorted(self.cells)],
            contact_clip=1e-6, interpolation=False, evaluation_target_fallback=False)
