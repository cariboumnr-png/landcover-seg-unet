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
Top-level namespace for training module.

Exposes selected public functions via lazy resolution to keep import
order simple and circular-free.
'''

from __future__ import annotations
import importlib
import typing

__all__ = [
    # classes
    'CallbackSet',
    'DataLoaders',
    'HeadSpecs',
    'HeadLosses',
    'HeadMetrics',
    'Optimization',
    # functions
    'build_callbacks',
    'build_dataloaders',
    'build_headspecs',
    'build_headlosses',
    'build_headmetrics',
    'build_optimization',
    'build_trainer_components',
]
# for static check
if typing.TYPE_CHECKING:
    from .callback import CallbackSet, build_callbacks
    from .dataloading import DataLoaders, build_dataloaders
    from .heads import HeadSpecs, build_headspecs
    from .loss import HeadLosses, build_headlosses
    from .metrics import HeadMetrics, build_headmetrics
    from .optimization import Optimization, build_optimization
    from .factory import build_trainer_components

def __getattr__(name: str):

    if name in ['CallbackSet', 'build_callbacks']:
        return getattr(importlib.import_module('.callback', __package__), name)
    if name in ['DataLoaders', 'build_dataloaders']:
        return getattr(importlib.import_module('.dataloading', __package__), name)
    if name in ['HeadSpecs', 'build_headspecs']:
        return getattr(importlib.import_module('.heads', __package__), name)
    if name in ['HeadLosses', 'build_headlosses']:
        return getattr(importlib.import_module('.loss', __package__), name)
    if name in ['HeadMetrics', 'build_headmetrics']:
        return getattr(importlib.import_module('.metrics', __package__), name)
    if name in ['Optimization', 'build_optimization']:
        return getattr(importlib.import_module('.optimization', __package__), name)
    if name in ['build_trainer_components']:
        return getattr(importlib.import_module('.factory', __package__), name)

    raise AttributeError(name)
