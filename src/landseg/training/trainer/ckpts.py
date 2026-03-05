# =========================================================================== #
#           Copyright (c) His Majesty the King in right of Ontario,           #
#         as represented by the Minister of Natural Resources, 2026.          #
#                                                                             #
#                      © King's Printer for Ontario, 2026.                    #
#                                                                             #
#       Licensed under the Apache License, Version 2.0 (the 'License');       #
#          you may not use this file except in compliance with the            #
#                                  License.                                   #
#                  You may obtain a copy of the License at:                   #
#                                                                             #
#                  http://www.apache.org/licenses/LICENSE-2.0                 #
#                                                                             #
#    Unless required by applicable law or agreed to in writing, software      #
#     distributed under the License is distributed on an 'AS IS' BASIS,       #
#      WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or        #
#                                   implied.                                  #
#       See the License for the specific language governing permissions       #
#                       and limitations under the License.                    #
# =========================================================================== #

# pylint: disable=missing-function-docstring
'''Standalone basic model checkpointing functions.'''

# standard imports
import typing
# third-party imports
import torch
# local imports
import landseg.training.common as common

# checkpoint metadata
class CheckpointMetaLike(typing.TypedDict):
    '''Checkpont metadata'''
    metric: float
    epoch: int
    step: int

# publich functions
def save(
        model: common.MultiheadModelLike,
        ckpt_meta: CheckpointMetaLike,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler.LRScheduler | None,
        fpath: str
    ) -> None:
    '''
    Save model/optimizer/scheduler states.
    '''

    # save states to file
    state = {
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict() if scheduler is not None else None,
        'metric': ckpt_meta['metric'],
        'epoch': ckpt_meta['epoch'],
        'step': ckpt_meta['step']
    }
    torch.save(state, fpath)

def load(
        model: common.MultiheadModelLike,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler.LRScheduler | None,
        fpath: str,
        device: str
    ) -> CheckpointMetaLike:
    '''
    Load previously saved state dicts for model, optimizer,
    scheduler, return a meta dict.
    '''

    # load states dict from file
    checkpoint = torch.load(fpath, map_location=device)

    # load model first
    model.load_state_dict(checkpoint['model'])

    # load optimizer and scheduler if present
    if checkpoint.get('optimizer'):
        optimizer.load_state_dict(checkpoint['optimizer'])
    if checkpoint.get('scheduler') and scheduler is not None:
        scheduler.load_state_dict(checkpoint['scheduler'])

    # return meta dict
    return {
        'metric': checkpoint.get('metric', -float('inf')),
        'epoch': checkpoint.get('epoch', 0),
        'step': checkpoint.get('step', 0)
    }
