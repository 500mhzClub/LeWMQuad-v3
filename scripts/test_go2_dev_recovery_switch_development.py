"""Check the recovery switch (development).

--recovery on must compose exactly the runtime the full explicit fix list composed (same mixin
classes, same order, same MRO, for the C1/C3/C4 and C2 runtimes), so recovery-on behaviour is
unchanged. --recovery off keeps only the pose-loss record, which changes no decision.
"""
from lewm.dev_harness_fixes_development import DIAGNOSTIC_FIXES, compose, fixes_for
from lewm.navigation_capability_completed_support_development import CompletedSupportRuntimeMixin
from scripts import run_go2_dense_horizon_navigation_development as source

FULL = ['terminal', 'latch', 'deadlock', 'pose', 'stall', 'backup']


def main():
    assert fixes_for('on') == sorted(FULL), fixes_for('on')
    assert fixes_for('off') == list(DIAGNOSTIC_FIXES) == ['pose']
    for base in (source.DenseNavigationRuntime, source.DenseReactiveNavigationRuntime):
        on = type('On', (compose(fixes_for('on'), CompletedSupportRuntimeMixin), base), {})
        full = type('Full', (compose(FULL, CompletedSupportRuntimeMixin), base), {})
        assert on.__mro__[1].__bases__ == full.__mro__[1].__bases__
        assert on.__mro__[2:] == full.__mro__[2:]
        off = type('Off', (compose(fixes_for('off'), CompletedSupportRuntimeMixin), base), {})
        frozen = set(base.__mro__) | set(CompletedSupportRuntimeMixin.__mro__)
        extra = [k.__name__ for k in off.__mro__[2:] if k not in frozen]
        assert extra == ['PoseLossRecordMixin'], extra
    print('recovery on = full fix list (identical MRO for both runtimes); recovery off = pose record only')


if __name__ == '__main__':
    main()
