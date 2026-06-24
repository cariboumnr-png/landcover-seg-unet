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
Top-level namespace for `landseg.adapters.api.configurators`.

Exposes selected public functions via lazy resolution to keep import
order simple and circular-free.
'''
from __future__ import annotations
import importlib
import typing

__all__ = [
    # classes
    'BaseConfigurator',
    'DataIngestionConfigurator',
    'DataPreparationConfigurator',
    'TrainingSessionConfigurator',
    'StudySweepConfigurator'
    # functions
    # types
]

# for static check
if typing.TYPE_CHECKING:
    from .base import BaseConfigurator
    from .data_ingest import DataIngestionConfigurator
    from .data_prepare import DataPreparationConfigurator
    from .model_train import TrainingSessionConfigurator
    from .study_sweep import StudySweepConfigurator

def __getattr__(name: str):

    if name in {'BaseConfigurator'}:
        return getattr(importlib.import_module('.base', __package__), name)

    if name in {'DataIngestionConfigurator'}:
        return getattr(importlib.import_module('.data_ingest', __package__), name)

    if name in {'DataPreparationConfigurator'}:
        return getattr(importlib.import_module('.data_prepare', __package__), name)

    if name in {'TrainingSessionConfigurator'}:
        return getattr(importlib.import_module('.model_train', __package__), name)

    if name in {'StudySweepConfigurator'}:
        return getattr(importlib.import_module('.study_sweep', __package__), name)

    raise AttributeError(f'module {__name__!r} has no attribute {name!r}')
