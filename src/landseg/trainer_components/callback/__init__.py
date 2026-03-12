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
Top-level namespace for training.callback.

Exposes selected public functions via lazy resolution to keep import
order simple and circular-free.
'''

from __future__ import annotations
import importlib
import typing

__all__ = [
    # classes
    'Callback',
    'CallbackSet',
    'LoggingCallback',
    'TrainCallback',
    'ValCallback',
    'InferCallback',
    'ProgressCallback',
    # functions
    'build_callbacks'
]

# for static check
if typing.TYPE_CHECKING:
    from .base import Callback
    from .factory import CallbackSet, build_callbacks
    from .logging import LoggingCallback
    from .phase_infer import InferCallback
    from .phase_train import TrainCallback
    from .phase_val import ValCallback
    from .progress import ProgressCallback

def __getattr__(name: str):

    if name in ['Callback']:
        return getattr(importlib.import_module('.base', __package__), name)
    if name in ['CallbackSet', 'build_callbacks']:
        return getattr(importlib.import_module('.factory', __package__), name)
    if name in ['LoggingCallback']:
        return getattr(importlib.import_module('.logging', __package__), name)
    if name in ['InferCallback']:
        return getattr(importlib.import_module('.phase_infer', __package__), name)
    if name in ['TrainCallback']:
        return getattr(importlib.import_module('.phase_train', __package__), name)
    if name in ['ValCallback']:
        return getattr(importlib.import_module('.phase_val', __package__), name)
    if name in ['ProgressCallback']:
        return getattr(importlib.import_module('.progress', __package__), name)

    raise AttributeError(name)
