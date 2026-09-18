"""Evaluate the 300-feature tracker through the complete independent round trip."""
from pathlib import Path

from lewm.eligible_floor_registration_development import bind
from scripts import evaluate_feature_budget_300_development as source

OUTPUT=source.source.BASE/'go2_feature_budget300_full_journey_v1_attempt_001'
_write=bind(source.write,OUTPUT=OUTPUT)


def write(name,value):
    if name=='launch.json':
        value=value|dict(complete_recorded_independent_round_trip=True,
            final_observation_is_recorded_return_arrival=True,
            wrapper_source_sha256=source.source.source.digest(Path(__file__)))
    if name in ('result.json','partial_result.json'):
        value=value|dict(status='FEATURE_BUDGET300_FULL_JOURNEY_COMPLETE' if value['failure'] is None
            else 'FEATURE_BUDGET300_FULL_JOURNEY_FAILED',complete_recorded_independent_round_trip=value['failure'] is None)
    _write(name,value)


main=bind(source.main,OUTPUT=OUTPUT,COUNT=3440,write=write)


if __name__=='__main__':main()
