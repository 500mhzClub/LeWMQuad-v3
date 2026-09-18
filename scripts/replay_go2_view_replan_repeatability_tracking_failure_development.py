"""Replay the retained fourth assignment through the unchanged pose tracker."""
from contextlib import redirect_stdout
from lewm.eligible_floor_registration_development import bind
from scripts import run_go2_view_replan_repeatability_development as run
from scripts import replay_go2_cached_connectivity_tracking_failure_development as replay


def main():
    root=run.BASE/run.root_name(4)
    with (root/'tracking_failure_replay_stdout.log').open('x') as stream, redirect_stdout(stream):
        bind(replay.main,BASE=run.BASE,ROOT=root.name)()
    print(root/'tracking_failure_replay_v1/result.json',flush=True)


if __name__=='__main__':main()
