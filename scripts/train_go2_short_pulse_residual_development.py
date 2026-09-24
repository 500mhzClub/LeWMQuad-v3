"""Three matched residual fits with the fixed short-pulse augmented schedule."""
from lewm.eligible_floor_registration_development import bind
from lewm.nominal_motion_residual_learning_development import NominalResidualTrainer,PARAMETERIZATION
from scripts import train_go2_pre_switch_successor_development as worker
from scripts import nominal_motion_residual_snapshot_development as snapshot
from scripts import prepare_go2_short_pulse_training_development as data

OUTPUT=data.BASE/'go2_short_pulse_residual_matched_fits_v1_attempt_001'


def main():
    bind(worker.main,OUTPUT=OUTPUT,write=bind(worker.write,OUTPUT=OUTPUT),EXPECTED_CONTEXTS=4694,
        SCHEDULE=data.OUTPUT/'schedule.json',load_training_rows=data.load_training_rows,prepare=data.prepare,
        ObservationHorizonTrainer=NominalResidualTrainer,SCHEMA=snapshot.SCHEMA,
        validate_payload=snapshot.validate_payload,load_snapshot=snapshot.load_snapshot,
        MODEL_DEFINITION=dict(motion_parameterization=PARAMETERIZATION,
            motion_head_initialization='zero_xy_zero_sin_unit_cos_residual',
            loss_on_composed_absolute_outcomes=True,learned_rgb_latents_retained=True),
        MODEL_SOURCES=(__file__,'scripts/prepare_go2_short_pulse_training_development.py',
            'lewm/nominal_motion_residual_learning_development.py',
            'scripts/nominal_motion_residual_snapshot_development.py'))()


if __name__=='__main__':main()
