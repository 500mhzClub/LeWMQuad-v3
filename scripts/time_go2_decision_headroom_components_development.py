"""Bounded component timing only; decoded values and method choices are discarded."""
import copy
import json
from pathlib import Path
import time

import numpy as np
import torch
from torch.nn import functional as F

from lewm.dense_native_observation_development import dense_native_context
from lewm.dense_visual_motion_readout_development import pool_tokens
from scripts.run_go2_decision_headroom_branches_development import load_bound, save
from scripts import run_go2_maze_view_readout_recovery_development as fitted


@torch.inference_mode()
def run(model, source_root, budget, case):
    states = json.loads((source_root/'snapshots.json').read_text())
    if not states:
        save(source_root/'component_timing_unavailable.json', dict(reason='NO_CAPTURED_SOURCE_PACKET'))
        return
    state = states[0]
    state_root = source_root/f"state_{state['frame']:04d}"
    packet = load_bound(state_root/'decision.pkl', state['decision'])
    device = next(model.predictor.parameters()).device
    native = dense_native_context(packet['native_context'], observed_ns=packet['measured_ns'])
    pixels = native['pixels'].to(device)
    actions = torch.as_tensor(np.asarray(packet['candidate_applied_commands'])[:, :, [0,2]],
        dtype=torch.float32, device=device)
    controls = native['past_applied_commands'][:, [0,2]].reshape(3,5,2).to(device)
    controls = ((controls-model.control_mean)/model.control_std)[None].expand(6,-1,-1,-1)
    mask = torch.ones(6,768,dtype=torch.bool,device=device)
    horizons = torch.full((6,),8,dtype=torch.long,device=device)
    checkpoint = fitted.OUTPUT/'maze_data_final.pt'
    completed = json.loads((fitted.OUTPUT/'result.json').read_text())
    digest = fitted.original.digest(checkpoint)
    if completed['status'] != 'COMPLETE' or completed['checkpoint_sha256']['maze_data'] != digest:
        raise ValueError('timing requires the existing fixed-final maze-data head')
    weights = torch.load(checkpoint, map_location='cpu', weights_only=False)
    second = copy.deepcopy(model.readout)
    second.load_state_dict(weights['model_state_dict'])
    second.eval().requires_grad_(False)
    heads = dict(old_data=model.readout, maze_data=second)
    rows = []

    def measured(kind, repeat, function, **identity):
        budget.reserve_component(kind, case, repeat)
        torch.cuda.synchronize()
        start = time.perf_counter()
        value = function()
        torch.cuda.synchronize()
        rows.append(dict(component=kind, repeat=repeat, wall_s=time.perf_counter()-start, **identity))
        if not torch.isfinite(value).all():
            raise ValueError('nonfinite timing output')
        budget.check('component_timing')
        return value

    context = None
    for repeat, count in enumerate((3,1)):
        encoded = measured('encoder', repeat, lambda:model.encoder.tokens(pixels[-count:]), image_count=count)
        if count == 3:
            context = F.layer_norm(encoded.float(), (1024,))[None]
        predicted = measured('predictor', repeat, lambda:model.predictor(
            context.expand(6,-1,-1,-1), actions, horizons, mask, control=controls),
            candidate_count=6, horizon_ms=800)
        current = pool_tokens(context[:, -1]).expand(6,-1,-1)
        future = pool_tokens(F.layer_norm(predicted.float(), (1024,)))
        for name, head in heads.items():
            decoded = measured('readout_'+name, repeat, lambda:head(current, future), candidate_count=6)
            del decoded
        del encoded, predicted, current, future
    save(source_root/'component_timing.json', dict(rows=rows,
        fixed_head_sha256=dict(old_data=model.readout_identity['sha256'], maze_data=digest),
        first_fixed_source_snapshot=state['frame'], encoder_batch_sizes=[3,1],
        predictor_timing_is_full_six_candidate_800ms_batch=True,
        source_controller_calls_are_separate_from_extra_timing_passes=True,
        decoded_outputs_discarded=True, dense_features_persisted=False,
        method_selections_or_rankings_computed=False, phase2_authorized=False))
