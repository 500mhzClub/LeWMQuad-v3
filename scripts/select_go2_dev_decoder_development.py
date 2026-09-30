"""Apply the committed decoder selection rule to development decoder fits (30 Sep 2026).

Rule (docs/CURRENT_RESEARCH_BRIEF.md, commit 54944c4d, fixed before any phase-1 result):
- choose on eval_onpolicy (held-out closed-loop C1 decisions, layouts 16-21);
- score = the C3 decoder's median 800-ms XY error averaged over the six movement types; lower is
  better; scores within 1 mm are split by the mean |log(median ratio)| over the moving types;
- between input variants, (a) past_frames is preferred if within 3 mm of (b) history;
- report on eval_transfer (700 ms), eval_offline and eval_fresh_c3, never used for choosing;
- C4 is reported alongside and plays no part in the choice.

Usage: select_go2_dev_decoder_development.py NAME [NAME ...]
"""
import argparse
import json
import math
from pathlib import Path

from scripts.fit_go2_dev_decoder_development import CATEGORIES, OUT

MOVING = ('rest_start', 'turn', 'cruise', 'arc_steady', 'switch')
CHOICE_KEY = 'eval_onpolicy@800ms'
REPORT_KEYS = ('eval_transfer@700ms', 'eval_offline@800ms', 'eval_offline@500ms', 'eval_fresh_c3@800ms')
TIE_MM, VARIANT_MARGIN_MM = 1., 3.


def score(result, model='C3'):
    cells = result['eval'][f'{CHOICE_KEY}/{model}']
    errors = [cells[c]['median_xy_mm'] for c in CATEGORIES if c in cells]
    ratios = [abs(math.log(cells[c]['median_ratio'])) for c in MOVING if c in cells and cells[c]['median_ratio']]
    return dict(score_mm=sum(errors)/len(errors), types_scored=len(errors), mean_abs_log_ratio=sum(ratios)/len(ratios) if ratios else None)


def choose(results):
    scored = sorted(((score(r), r['name']) for r in results), key=lambda x: x[0]['score_mm'])
    best = scored[0]
    tied = [s for s in scored if s[0]['score_mm']-best[0]['score_mm'] <= TIE_MM]
    return min(tied, key=lambda s: s[0]['mean_abs_log_ratio'] if s[0]['mean_abs_log_ratio'] is not None else math.inf)[1], scored


def row(result, key, model):
    cells = result['eval'].get(f'{key}/{model}')
    if not cells:
        return None
    parts = []
    for c in ('all',)+CATEGORIES:
        if c in cells:
            x = cells[c]
            ratio = '-' if x['median_ratio'] is None else f"{x['median_ratio']:.2f}"
            parts.append(f"{c} {ratio}·{x['median_xy_mm']:.0f}")
    return ' | '.join(parts)


def main(names):
    results = [json.loads((OUT/f'{n}.json').read_text()) for n in names]
    chosen, scored = choose(results)
    print(f'Choice set {CHOICE_KEY} (C3): mean over movement types of median XY error (mm)')
    for s, name in scored:
        tag = '  <- chosen' if name == chosen else ''
        ratio = '-' if s['mean_abs_log_ratio'] is None else f"{s['mean_abs_log_ratio']:.3f}"
        print(f"  {name:32s} {s['score_mm']:6.1f} mm  (types {s['types_scored']}, mean|log ratio| {ratio}){tag}")
    print('\nReport sets (not used for choosing): median ratio · median XY error (mm) by movement type')
    for key in (CHOICE_KEY,)+REPORT_KEYS:
        print(f'\n{key}')
        for r in results:
            for model in ('C3', 'C4'):
                text = row(r, key, model)
                if text:
                    print(f"  {r['name']:32s} {model}  {text}")


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('names', nargs='+')
    main(p.parse_args().names)
