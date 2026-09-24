"""Eight one-command blocks, each one actual 100-ms control interval."""
import torch
from lewm.geometry_progress_pilot_development import candidate_commands


def plan(action,*,offset_ticks=0):
    if type(offset_ticks) is not int or offset_ticks not in range(0,40,5):
        raise ValueError('exact existing family context offset required')
    commands=candidate_commands(action)[offset_ticks:offset_ticks+8]
    blocks=torch.zeros((8,1,3),dtype=torch.float32)
    valid=torch.zeros((8,1),dtype=torch.bool)
    blocks[:len(commands),0]=torch.tensor(commands,dtype=torch.float32)/torch.tensor([.3,1.,.5])
    valid[:len(commands),0]=True
    return blocks,valid


def validate_plan(blocks,valid,batch):
    if blocks.shape!=(batch,8,1,3) or valid.shape!=(batch,8,1) or valid.dtype!=torch.bool:
        raise ValueError('B,8,1,3 commands and boolean B,8,1 validity required')
    if (batch<1 or blocks.device!=valid.device or not blocks.is_floating_point()
            or not torch.isfinite(blocks).all() or torch.count_nonzero(blocks[~valid])
            or torch.count_nonzero(blocks[...,1]) or (blocks.abs()>1+1e-6).any()):
        raise ValueError('finite normalized fixed-axis command prefix and zero unknown padding required')
    active=valid[...,0]
    if not active[:,0].all() or (active[:,1:]&~active[:,:-1]).any():
        raise ValueError('nonempty contiguous known actual command prefix required')
    offsets=active.to(torch.int64).cumsum(-1)*100_000_000
    return active,torch.where(active,offsets,torch.zeros_like(offsets))
