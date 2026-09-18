"""Prequential, public-observation-only residual diagnostic; no controller."""
from collections import deque
import numpy as np

WINDOW_TICKS = 8


def replay_bias(records):
    """Apply the preceding eight ticks' mean XY residual, then reveal this target.

    A tick-t outcome arrives at t+1. Native outcomes are not accepted here.
    Missing ticks reduce the available sample count; no target is imputed.
    This is an offline causal diagnostic, not adopted prediction correction.
    """
    history = deque(); output = []; previous = -1
    for r in records:
        if set(r) != {'tick', 'available_tick', 'predicted_body_xy_m', 'observed_body_xy_m'}:
            raise ValueError('only explicit public-observation residual fields accepted')
        tick = r['tick']; available = r['available_tick']
        p = np.asarray(r['predicted_body_xy_m'], float)
        y = np.asarray(r['observed_body_xy_m'], float)
        if (type(tick) is not int or tick <= previous or type(available) is not int
                or available != tick+1 or p.shape != (2,) or y.shape != (2,)
                or not np.isfinite([p, y]).all()):
            raise ValueError('ordered exact one-step public observations required')
        while history and tick-history[0][0] > WINDOW_TICKS: history.popleft()
        bias = np.mean([v for _, _, v in history], axis=0) if history else np.zeros(2)
        if any(t >= tick or when > tick for t, when, _ in history):
            raise ValueError('future observed residual cannot enter current score')
        output.append(dict(tick=tick, correction_xy_m=bias.tolist(),
            corrected_body_xy_m=(p-bias).tolist(),
            residual_source_ticks=[t for t, _, _ in history],
            residual_available_ticks=[when for _, when, _ in history],
            observed_residual_samples=len(history), native_outcomes_used=False))
        # Reveal the actual current outcome only after the current prediction.
        history.append((tick, available, p-y)); previous = tick
    return output
