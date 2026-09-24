"""Describe survey action restrictions and recovery triggers in saved plans."""
import argparse
from collections import Counter
import json
from statistics import median

from scripts.navigation_artifact_root_development import BASE, validate_root


def stats(values):
    return (dict(count=len(values), minimum=min(values), median=median(values),
                 maximum=max(values)) if values else dict(count=0))


def diagnose(root, start, end):
    validate_root(root)
    plans = json.loads((root/'planning.json').read_text())
    plans = [p for p in plans if start <= p['frame'] <= end
             and 'scan_utilities' in p.get('selection', {})]
    counts = Counter(); margins = []; rows = []
    for p in plans:
        s = p['selection']
        candidates = {r['action']: r for r in s['memory_forecast_candidates']}
        preferred = s['before_memory_filter_action']
        c = candidates[preferred]
        translations = [a for a in ('forward', 'left_arc', 'right_arc')
                        if candidates[a]['full_reserve_path_clear']]
        allowed = {r['action'] for r in s['scan_utilities']}
        excluded = [a for a in translations if a not in allowed]
        latch = s.get('clearance_turn') or {}
        new_latch = latch.get('event') == 'CLEAR_ALTERNATIVE_TURN_LATCHED'
        opposite = (preferred in ('left_turn', 'right_turn')
                    and p['action'] in ('left_turn', 'right_turn')
                    and preferred != p['action'])
        counts['plans'] += 1
        counts['on_time'] += bool(p['on_time'])
        counts['full_reserve_translation_excluded_from_scan_scoring'] += bool(excluded)
        counts['opposite_turn_selected'] += opposite
        counts['opposite_turn_with_excluded_clear_translation'] += opposite and bool(excluded)
        counts['new_alternative_latch'] += new_latch
        if new_latch:
            counts['new_latch_preferred_nominal_footprint_clear'] += c['nominal_footprint_path_clear']
            counts['new_latch_with_excluded_clear_translation'] += bool(excluded)
            minimum = c['minimum_predicted_path_clearance_m']
            if minimum is not None:
                margins.append(minimum-c['required_path_clearance_m'])
        rows.append(dict(frame=p['frame'], action=p['action'], on_time=p['on_time'],
            preferred=preferred, new_latch=new_latch,
            preferred_minimum_clearance_m=c['minimum_predicted_path_clearance_m'],
            preferred_required_clearance_m=c['required_path_clearance_m'],
            preferred_nominal_footprint_clear=c['nominal_footprint_path_clear'],
            excluded_full_reserve_translations=excluded))
    return dict(root_name=root.name, start_frame=start, end_frame=end,
        counts=dict(counts), selected_actions=dict(Counter(p['action'] for p in plans)),
        new_latch_preferred_reserve_margin_m=stats(margins),
        interpretation='Saved forecast eligibility only; excluded translations have no survey utility or counterfactual execution evidence.',
        thresholds_changed=False, controller_changed=False, forecast_recomputed=False,
        counterfactual_success_established=False, rows=rows)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root-name', required=True)
    parser.add_argument('--start-frame', type=int, default=2000)
    parser.add_argument('--end-frame', type=int, default=4000)
    args = parser.parse_args()
    root = BASE/args.root_name
    result = diagnose(root, args.start_frame, args.end_frame)
    target = root/f'saved_scan_recovery_{args.start_frame}_{args.end_frame}_v1.json'
    with target.open('x') as f:
        json.dump(result, f, indent=2)
    print(json.dumps({k:v for k,v in result.items() if k != 'rows'}))
