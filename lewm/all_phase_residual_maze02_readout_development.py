"""Evaluator-only compact progress, contact and timing accounting."""
import numpy as np


def traversal_counts(traversal):
    rows=[] if traversal is None else traversal['crossings']
    edges={tuple(sorted((tuple(r['from_cell']),tuple(r['to_cell']))))
        for r in rows if r['declared_open_edge']}
    return dict(total_crossings=len(rows),distinct_open_edges=len(edges),
        invalid_crossings=sum(not r['declared_open_edge'] for r in rows),
        distinct_open_edge_cells=[[list(a),list(b)] for a,b in sorted(edges)])


def timing(values):
    a=np.asarray(values,float)
    if a.ndim!=1 or not len(a) or not np.isfinite(a).all() or (a<0).any():
        raise ValueError('complete finite nonnegative observed timing required')
    return dict(samples=len(a),median_ms=float(np.median(a)),p95_ms=float(np.quantile(a,.95)),
        maximum_ms=float(a.max()),samples_above_command_interval_100ms=int((a>100).sum()))


def case_readout(report,collection,physics_contact):
    contact=np.asarray(physics_contact)
    if contact.ndim!=1 or len(contact)<900 or not np.isin(contact,[0,1]).all():
        raise ValueError('complete binary native physics contact evidence required')
    evaluation=report['native_evaluation'];mission=collection['mission_receipt']
    return dict(schedule_terminal=collection['schedule_terminal'],physical_stop=collection['physical_stop'],
        acquisition_stop=collection['acquisition_stop'],terminal_zero_ticks=collection['terminal_zero_ticks'],
        observed_arrivals=[] if mission is None else mission['arrivals'],
        native_arrival_windows=evaluation['arrival_windows'],
        outbound=traversal_counts(evaluation['outbound_traversal']),
        returning=traversal_counts(evaluation['return_traversal']),
        physically_retraced_outbound_route=evaluation['physically_retraced_outbound_route'],
        native_physics_samples=len(contact),native_contact_samples=int(np.count_nonzero(contact)),
        native_contact_free=bool(not contact.any()),
        observation_and_control=timing(report['observation_and_control_wall_ms']),
        iteration_with_receipt=timing(report['iteration_with_receipt_wall_ms']),
        strict_physical_visibility_pass=report['strict_physical_visibility_pass'],
        hard_measurement_failed_frames=report['hard_measurement_failed_frames'],
        verified_round_trip=report['verified_round_trip'],evaluator_only=True,real_time_qualified=False)
