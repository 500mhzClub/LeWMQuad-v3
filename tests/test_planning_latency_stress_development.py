from threading import Condition, Event, Thread, local

from lewm.planning_latency_stress_development import PlanningLatencyClock


def test_planning_delay_waits_for_physics_and_applies_once():
    entered = Event()

    class Base:
        def __init__(self):
            self.local = local()
            self.ns = 100_000_000
            self.closed = False
            self.condition = Condition()

        def begin(self, stage):
            self.local.stage = stage
            self.local.start_sim = self.ns
            self.local.waited = 0

        def end(self):
            self.local.stage = None

        def __call__(self):
            entered.set()
            return self.ns

    base = Base()
    clock = PlanningLatencyClock(base)
    done = Event()
    results = []

    def work():
        clock.begin('planning')
        results.extend([clock(), clock()])
        clock.end()
        done.set()

    thread = Thread(target=work)
    thread.start()
    try:
        assert entered.wait(1) and not done.wait(.01)
        with base.condition:
            base.ns = 119_000_000
            base.condition.notify_all()
        assert not done.wait(.01)
        with base.condition:
            base.ns = 120_000_000
            base.condition.notify_all()
        assert done.wait(1)
        assert results == [120_000_000, 120_000_000]
        assert len(clock.rows) == 1
        assert clock.rows[0]['earliest_release_ns'] == 120_000_000
        clock.begin('tracking')
        assert clock() == 120_000_000
        clock.end()
        assert len(clock.rows) == 1
    finally:
        with base.condition:
            base.closed = True
            base.condition.notify_all()
        thread.join(timeout=1)
