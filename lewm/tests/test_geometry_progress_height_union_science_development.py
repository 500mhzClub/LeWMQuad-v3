"""New aggregation preserves full denominators and distinct measurement criteria."""
from copy import deepcopy
import pytest
from lewm.tests.test_geometry_progress_science_development import reports as old_reports
from lewm.geometry_progress_near_field_development import measurement_gate
from scripts.read_go2_geometry_progress_height_union_science_v1 import summarize


def reports():
    rows=old_reports()
    for r in rows:
        r.update(hard_measurement_failed_frames=[],strict_physical_visibility_pass=True,
            footprint_checks=[dict(stable_interior_metric_pass=True,near_occlusion_failure=False) for _ in range(r['frames'])])
    return rows


def test_complete_native_counts_and_modality_masks_are_preserved():
    rows=reports();r=summarize(rows,measurement_gate(rows))
    assert r['design_and_measurement_gate_pass'] and r['successful_progress']==4
    assert r['contact_episodes']==8 and r['target_accounting']['recorded_slots']==192
    assert r['measurement_accounting']['footprint_frames']==48
    assert r['target_accounting']['positive_with_missing_image']==64
    assert not r['navigation_qualified'] and not r['model_trained']


def test_prospective_footprint_contract_does_not_relabel_strict_failure():
    rows=reports();rows[0]['depth_checks'][0]['within1mm']=False
    rows[0]['strict_physical_visibility_pass']=False
    r=summarize(rows,measurement_gate(rows))
    assert r['design_and_measurement_gate_pass']
    assert not r['legacy_strict_depth_and_design_criterion']
    assert r['measurement_accounting']['strict_failed_cases']==[rows[0]['trial']]
    assert not r['measurement_accounting']['depth_navigation_qualified']


@pytest.mark.parametrize('fault',['stable','occlusion'])
def test_native_design_cannot_override_hard_measurement_failure(fault):
    rows=reports();rows[0]['hard_measurement_failed_frames']=[0]
    rows[0]['footprint_checks'][0].update(stable_interior_metric_pass=fault!='stable',near_occlusion_failure=fault=='occlusion')
    r=summarize(rows,measurement_gate(rows))
    assert r['panel']['informative_for_next_dataset'] and not r['design_and_measurement_gate_pass']


def test_audited_gate_cannot_be_replaced_or_subset():
    rows=reports();gate=measurement_gate(rows);bad=deepcopy(gate)
    bad['prediction_design_and_measurement_gate_pass']=False
    with pytest.raises(ValueError,match='gate mismatch'):summarize(rows,bad)
    with pytest.raises(ValueError,match='complete ordered'):summarize(rows[:-1],gate)
