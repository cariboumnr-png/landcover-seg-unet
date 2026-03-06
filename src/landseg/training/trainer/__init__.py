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

'''
Top-level namespace for training.trainer.

Exposes selected public functions via lazy resolution to keep import
order simple and circular-free.
'''

from __future__ import annotations
import importlib
import typing

__all__ = [
    # classes
    'MultiHeadTrainer',
    'TrainerComponents',
    'RuntimeConfig',
    'RuntimeState',
    # functions
    'get_config',
    'init_trainer_state',
    'load',
    'save',
    'multihead_loss',
]

# for static check
if typing.TYPE_CHECKING:
    from .ckpts import load, save
    from .comps import TrainerComponents
    from .config import RuntimeConfig, get_config
    from .loss import multihead_loss
    from .state import RuntimeState, init_trainer_state
    from .trainer import MultiHeadTrainer

def __getattr__(name: str):

    if name in ['load', 'save']:
        return getattr(importlib.import_module('.ckpts', __package__), name)
    if name in ['TrainerComponents']:
        return getattr(importlib.import_module('.comps', __package__), name)
    if name in ['RuntimeConfig', 'get_config']:
        return getattr(importlib.import_module('.config', __package__), name)
    if name in ['multihead_loss']:
        return getattr(importlib.import_module('.loss', __package__), name)
    if name in ['RuntimeState', 'init_trainer_state']:
        return getattr(importlib.import_module('.state', __package__), name)
    if name in ['MultiHeadTrainer']:
        return getattr(importlib.import_module('.trainer', __package__), name)

    raise AttributeError(name)
