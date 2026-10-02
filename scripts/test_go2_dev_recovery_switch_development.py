"""Check the recovery switch (development).

--recovery on must compose exactly the runtime the full explicit fix list composed (same mixin
classes, same order, same MRO, for the C1/C3/C4 and C2 runtimes), so recovery-on behaviour is
unchanged. --recovery off keeps only the pose-loss record, which changes no decision.
"""
from lewm.dev_harness_fixes_development import DEFAULT_RECOVERY, compose, fixes_for
from lewm.navigation_capability_completed_support_development import CompletedSupportRuntimeMixin
from scripts import run_go2_dense_horizon_navigation_development as source

FULL = ['terminal', 'latch', 'deadlock', 'pose', 'stall', 'backup']


def main():
    assert DEFAULT_RECOVERY == 'off'
    # Preliminary-run sets (no harness fixes) are unchanged.
    assert fixes_for('on', harness=False) == sorted(FULL) and fixes_for('off', harness=False) == ['pose']
    # From 2 Oct the coverage-rule fix applies in both settings.
    assert fixes_for('on') == sorted(FULL+['coverage']) and fixes_for('off') == ['coverage', 'pose'], (fixes_for('on'), fixes_for('off'))
    for base in (source.DenseNavigationRuntime, source.DenseReactiveNavigationRuntime):
        on = type('On', (compose(fixes_for('on', harness=False), CompletedSupportRuntimeMixin), base), {})
        full = type('Full', (compose(FULL, CompletedSupportRuntimeMixin), base), {})
        assert on.__mro__[1].__bases__ == full.__mro__[1].__bases__
        assert on.__mro__[2:] == full.__mro__[2:]
        off = type('Off', (compose(fixes_for('off', harness=False), CompletedSupportRuntimeMixin), base), {})
        frozen = set(base.__mro__) | set(CompletedSupportRuntimeMixin.__mro__)
        extra = [k.__name__ for k in off.__mro__[2:] if k not in frozen]
        assert extra == ['PoseLossRecordMixin'], extra
    print('preliminary sets unchanged (on = full list, off = pose record only); default off; coverage fix in both settings from 2 Oct')


if __name__ == '__main__':
    main()
