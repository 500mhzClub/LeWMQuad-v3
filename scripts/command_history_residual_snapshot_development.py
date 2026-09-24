"""Snapshots explicitly identify the command-history reference and learned head."""
import torch
from lewm.command_history_residual_learning_development import CommandHistoryResidualTrainer, PARAMETERIZATION
from lewm.eligible_floor_registration_development import bind
from lewm.pulse_timed_training_runner_development import state_digest
from scripts import nominal_motion_residual_snapshot_development as previous
from scripts.observation_horizon_snapshot_development import config

SCHEMA='command_history_residual_snapshot.v1'


def validate_payload(payload,expected_binding,expected_config):
    saved=payload['trainer']
    if (payload['schema']!=SCHEMA or payload['binding']!=expected_binding
            or saved['schema']!='command_history_residual_training_development.v1'
            or saved['motion_parameterization']!=PARAMETERIZATION or saved['failed']
            or {k:saved[k] for k in expected_config}!=expected_config):
        raise ValueError('explicit command-history residual identity required')
    clone=CommandHistoryResidualTrainer(**{k:v for k,v in expected_config.items() if k!='updates'})
    if clone.initial_sha256!=saved['initial_sha256']:raise ValueError('initial reference/model identity changed')
    for key,value in clone.model.state_dict().items():
        if key.startswith('reference_') and not torch.equal(value,saved['model_state'][key]):
            raise ValueError('frozen reference changed during training')
    clone.model.load_state_dict(saved['model_state'],strict=True)
    if (state_digest(clone.model.state_dict())!=saved['model_sha256']
            or not all(torch.isfinite(p).all() for p in clone.model.state_dict().values())):
        raise ValueError('finite exact model state required')
    clone.updates=saved['updates'];clone.evaluation_only=True;clone.model.eval()
    return clone


load_snapshot=bind(previous.load_snapshot,validate_payload=validate_payload)
