import numpy as np
import pytest
from lewm.gyro_coherent_floor_constraint_development import fit_pair, constrain_translation


def test_tilted_translated_floor_recovers_gyro_consistent_height():
    rng = np.random.default_rng(2026091518)
    a = np.column_stack((rng.uniform(-1, 1, (400, 2)), np.full(400, -.3)))
    angle = .13
    G = np.array([[np.cos(angle), 0, np.sin(angle)], [0, 1, 0],
                  [-np.sin(angle), 0, np.cos(angle)]])
    translation = np.array([.04, -.02, .009])
    b = (a-translation)@G
    c = fit_pair(a, b, G, reference_pool_count=400, current_pool_count=400,
        reference_up=np.array([0.,0.,1.]), current_up=G.T@np.array([0.,0.,1.]),
        minimum_second_eigenvalue_m2=.0004)
    np.testing.assert_allclose(np.asarray(c['reference_normal_body']),
        G@np.asarray(c['current_normal_body']), atol=1e-14)
    np.testing.assert_allclose(constrain_translation(np.array([.04,-.02,0.]),c,G),
        translation, atol=1e-13)
    assert c['retained_counts'] == [400,400]


def test_collinear_floor_cannot_supply_two_axis_support():
    a = np.column_stack((np.linspace(-1,1,200),np.zeros(200),np.full(200,-.3)))
    with pytest.raises(ValueError, match='two-axis extent'):
        fit_pair(a,a,np.eye(3),reference_pool_count=200,current_pool_count=200,
            reference_up=np.array([0.,0.,1.]),current_up=np.array([0.,0.,1.]),
            minimum_second_eigenvalue_m2=.0004)
