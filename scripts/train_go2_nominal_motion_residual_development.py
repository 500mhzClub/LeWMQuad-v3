"""Same data/budget worker, explicitly different outcome parameterization."""
from lewm.eligible_floor_registration_development import bind
from lewm.nominal_motion_residual_learning_development import NominalResidualTrainer,PARAMETERIZATION
from scripts import train_go2_pre_switch_successor_development as previous
from scripts import nominal_motion_residual_snapshot_development as snapshot

OUTPUT=previous.BASE/'go2_nominal_motion_residual_matched_fits_v1_attempt_001'
WRITE=bind(previous.write,OUTPUT=OUTPUT)


def main():
    bind(previous.main,OUTPUT=OUTPUT,write=WRITE,ObservationHorizonTrainer=NominalResidualTrainer,
        SCHEMA=snapshot.SCHEMA,validate_payload=snapshot.validate_payload,load_snapshot=snapshot.load_snapshot,
        MODEL_DEFINITION=dict(motion_parameterization=PARAMETERIZATION,
            motion_head_initialization='zero_xy_zero_sin_unit_cos_residual',
            loss_on_composed_absolute_outcomes=True,learned_rgb_latents_retained=True),
        MODEL_SOURCES=(__file__,'lewm/nominal_motion_residual_learning_development.py',
            'scripts/nominal_motion_residual_snapshot_development.py'))()


if __name__=='__main__':main()
