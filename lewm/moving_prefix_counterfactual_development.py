"""Prepared off-diagonal moving-prefix collection; no runtime or source export."""
import copy

from lewm.counterfactual_maze_development import ACTIONS,corpus,branch_spec

MOVING_PREFIX_TICKS=10
SUFFIX_TICKS=30
RELEASE_TICKS=5


def trials():
    """384 new suffixes plus existing diagonal evidence, not 384 new layouts.

    The source dataset contains every selected one-second moving context.
    A collector must still reproduce and verify each physical/body/RGB prefix;
    these specs do not imply that reuse or execution has passed that check.
    """
    rows=[]
    for layout in corpus():
        for past in range(1,5):
            reference=branch_spec(layout,past)
            for future in range(5):
                if future==past: continue
                row=copy.deepcopy(layout)
                row.update(scene_id=f'moving-prefix-counterfactual-development-v1-{layout["layout_index"]:02d}-{ACTIONS[past][0]}-to-{ACTIONS[future][0]}',
                    family='MOVING_PREFIX_COUNTERFACTUAL_DEVELOPMENT',case_index=layout['layout_index'],arm='baseline',
                    prefix_action_index=past,prefix_action_name=ACTIONS[past][0],prefix_command=list(ACTIONS[past][1]),
                    future_action_index=future,future_action_name=ACTIONS[future][0],future_command=list(ACTIONS[future][1]),
                    reference_scene_id=reference['scene_id'],moving_prefix_ticks=MOVING_PREFIX_TICKS,
                    suffix_ticks=SUFFIX_TICKS,release_ticks=RELEASE_TICKS)
                rows.append(row)
    return rows


def evidence_cells():
    """600 planned context/action cells with explicit existing/new provenance."""
    fresh={(r['layout_id'],r['prefix_action_index'],r['future_action_index']):r['scene_id'] for r in trials()}
    cells=[]
    for layout in corpus():
        for past in range(5):
            for future in range(5):
                if past==0:
                    source=branch_spec(layout,future)['scene_id']; offset=0; provenance='existing_initial_branch'
                elif past==future:
                    source=branch_spec(layout,past)['scene_id']; offset=1_000_000_000; provenance='existing_moving_continuation'
                else:
                    source=fresh[(layout['layout_id'],past,future)]; offset=None; provenance='new_physical_switch_required'
                cells.append({'layout_id':layout['layout_id'],'data_role':layout['data_role'],'prefix_action_index':past,
                    'future_action_index':future,'provenance':provenance,'scene_id':source,'existing_offset_ns':offset,
                    'known_horizon_count':8 if past==0 else 6})
    return cells
