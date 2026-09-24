from collections import Counter

from lewm.moving_prefix_counterfactual_development import trials,evidence_cells


def test_complete_off_diagonal_moving_panel_without_new_layout_claim():
    rows=trials()
    assert len(rows)==384 and len({r['scene_id'] for r in rows})==384
    assert len({r['layout_id'] for r in rows})==24
    assert Counter(r['data_role'] for r in rows)=={'train':256,'validation':128}
    assert all(r['prefix_action_index']!=r['future_action_index'] and r['prefix_action_index']!=0 for r in rows)
    for layout in {r['layout_id'] for r in rows}:
        assert {(r['prefix_action_index'],r['future_action_index']) for r in rows if r['layout_id']==layout}=={
            (a,b) for a in range(1,5) for b in range(5) if a!=b}


def test_source_cells_are_explicit_not_synthetic_alternative_outcomes():
    cells=evidence_cells()
    assert len(cells)==600 and len({(r['layout_id'],r['prefix_action_index'],r['future_action_index']) for r in cells})==600
    assert Counter(r['provenance'] for r in cells)=={'existing_initial_branch':120,'existing_moving_continuation':96,'new_physical_switch_required':384}
    assert Counter(r['data_role'] for r in cells)=={'train':400,'validation':200}
    assert all(r['existing_offset_ns'] is None for r in cells if r['provenance']=='new_physical_switch_required')
    assert all(r['known_horizon_count']==6 for r in cells if r['prefix_action_index']!=0)


def test_specs_are_deterministic_and_do_not_share_mutable_geometry():
    rows=trials(); assert rows==trials()
    rows[0]['geometry']['wall_boxes'].clear()
    assert rows[1]['geometry']['wall_boxes']
    assert all(r['moving_prefix_ticks']==10 and r['suffix_ticks']==30 and r['release_ticks']==5 for r in rows)
