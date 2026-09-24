"""Matched-clock diagnostics, not a fusion algorithm or clearance contract."""
import numpy as np

SUPPORT_MODES = ('stationary_centre', 'level_sphere_rolling')


def stats(values):
    values = np.asarray(values, float)
    if values.ndim != 1 or not np.isfinite(values).all():
        raise ValueError('finite scalar observations required; missing values stay separate')
    return dict(count=len(values), mean=float(values.mean()) if len(values) else None,
                p95=float(np.quantile(values, .95)) if len(values) else None,
                maximum=float(values.max()) if len(values) else None)


def support_index(rows):
    times = [r['measured_ns'] for r in rows]
    if (not times or any(type(t) is not int for t in times)
            or any(b-a != 20_000_000 for a, b in zip(times, times[1:]))):
        raise ValueError('complete ordered 50Hz support observations required')
    for row in rows:
        values = [row['modes'][m]['consensus_velocity_body_m_s'] for m in SUPPORT_MODES]
        if (values[0] is None) != (values[1] is None):
            raise ValueError('fixed support hypotheses must share availability')
        for value in values:
            if value is not None and (np.asarray(value).shape != (3,) or not np.isfinite(value).all()):
                raise ValueError('finite velocity or explicit None required')
    return dict(zip(times, rows, strict=True))


def contact_window(index, start, end):
    if type(start) is not int or type(end) is not int or end-start != 100_000_000:
        raise ValueError('exact 100ms camera interval required')
    stamps = list(range(start, end+1, 20_000_000))
    if any(t not in index for t in stamps):
        raise ValueError('all six endpoint-inclusive support samples required')
    rows = [index[t] for t in stamps]
    missing = [t for t, row in zip(stamps, rows, strict=True)
               if row['modes']['level_sphere_rolling']['consensus_velocity_body_m_s'] is None]
    return dict(start_ns=start, end_ns=end, support_samples=6,
                unavailable_samples=len(missing), unavailable_measured_ns=missing,
                contact_complete=not missing)


def dropout_spans(index):
    """Inclusive sample ranges; no extrapolation of exact physical dropout times."""
    spans = []
    for stamp, row in index.items():
        if row['modes']['level_sphere_rolling']['consensus_velocity_body_m_s'] is not None:
            continue
        if not spans or stamp-spans[-1]['last_missing_ns'] != 20_000_000:
            spans.append(dict(first_missing_ns=stamp, last_missing_ns=stamp, missing_samples=1))
        else:
            spans[-1]['last_missing_ns'] = stamp
            spans[-1]['missing_samples'] += 1
    return spans


def displacement_error(before, after, true_before, true_after):
    """Translation difference in the shared initial body frame, in metres.

    No division by time or assumption of a stationary body during the interval.
    A terminal visual observation remains unavailable, never zero motion.
    """
    if before is None or after is None:
        return None
    predicted = np.asarray(after['position_initial_body_m'])-before['position_initial_body_m']
    truth = np.asarray(true_after)-true_before
    if predicted.shape != (3,) or truth.shape != (3,) or not np.isfinite([predicted, truth]).all():
        raise ValueError('finite three-dimensional displacements required')
    return float(np.linalg.norm(predicted-truth))


def summarize_windows(rows):
    result = {}
    for label, selected in (
        ('all', rows),
        ('contact_complete', [r for r in rows if r['contact_complete']]),
        ('contact_dropout', [r for r in rows if not r['contact_complete']]),
    ):
        errors = [r['displacement_error_m'] for r in selected if r['displacement_error_m'] is not None]
        result[label] = dict(windows=len(selected), visual_unavailable=len(selected)-len(errors),
                             displacement_error_m=stats(errors))
    return result
