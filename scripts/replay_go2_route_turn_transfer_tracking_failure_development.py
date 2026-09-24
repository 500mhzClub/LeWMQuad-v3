"""After the fixed pair, replay its first retained tracking failure unchanged."""
from contextlib import redirect_stdout

from lewm.eligible_floor_registration_development import bind
from scripts import run_go2_route_turn_memory_transfer_development as run
from scripts import replay_go2_cached_connectivity_tracking_failure_development as replay


if __name__ == '__main__':
    if not (run.BASE/run.root_name(2)/'frozen_readout_navigation_readout_v1.json').exists():
        raise ValueError('finish the fixed native pair before heavy replay')
    root=run.BASE/run.root_name(1)
    with (root/'tracking_failure_replay_stdout.log').open('x') as stream, redirect_stdout(stream):
        bind(replay.main,BASE=run.BASE,ROOT=root.name)()
    print(root/'tracking_failure_replay_v1/result.json',flush=True)
