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

# pylint: disable=too-many-return-statements

'''
Top-level namespace for `landseg.trainer_engine`.

Exposes selected public functions via lazy resolution to keep import
order simple and circular-free.
'''

from __future__ import annotations
import importlib
import typing

__all__ = [
    # classes
    'MultiHeadTrainer',
    'RuntimeConfig',
    'RuntimeState',
    # functions
    'export_previews',
    'get_config',
    'init_state',
    'load',
    'save',
    'multihead_loss',
]

# for static check
if typing.TYPE_CHECKING:
    from .engine_config import RuntimeConfig, get_config
    from .engine_state import RuntimeState, init_state
    from .engine import MultiHeadTrainer
    from .utils import load, save,  multihead_loss, export_previews

def __getattr__(name: str):

    if name in ['RuntimeConfig', 'get_config']:
        return getattr(importlib.import_module('.engine_config', __package__), name)
    if name in ['RuntimeState', 'init_state']:
        return getattr(importlib.import_module('.engine_state', __package__), name)
    if name in ['MultiHeadTrainer']:
        return getattr(importlib.import_module('.engine', __package__), name)
    if name in ['load', 'save',  'multihead_loss', 'export_previews']:
        return getattr(importlib.import_module('.utils', __package__), name)

    raise AttributeError(name)
