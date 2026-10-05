"""Tests for the C1R mixin (lewm/dev_c1_refit_development.py)."""
import numpy as np

from lewm.dev_c1_refit_development import RefitCommandModelMixin, load_refit
from lewm.short_pulse_navigation_runtime_development import COMMAND_FIT


class FakeRuntime:
    def __init__(self, *args, **kwargs):
        with np.load(COMMAND_FIT/'command_only.npz', allow_pickle=False) as a:
            self.command_model = {k: a[k].copy() for k in ('mean', 'scale', 'bias', 'coefficient')}
        self.command_fit_sha256 = 'deployed'


def test_refit_replaces_deployed_model():
    runtime = type('R', (RefitCommandModelMixin, FakeRuntime), {})()
    model, sha = load_refit()
    assert runtime.command_fit_sha256 == sha != 'deployed'
    for k in model:
        np.testing.assert_array_equal(runtime.command_model[k], model[k])
    with np.load(COMMAND_FIT/'command_only.npz') as a:
        assert not np.array_equal(runtime.command_model['coefficient'], a['coefficient'])


def test_requires_command_model():
    class Bare:
        def __init__(self):
            pass
    try:
        type('R', (RefitCommandModelMixin, Bare), {})()
    except ValueError:
        return
    raise AssertionError('missing command model must be refused')


if __name__ == '__main__':
    test_refit_replaces_deployed_model()
    test_requires_command_model()
    print('2 passed')
