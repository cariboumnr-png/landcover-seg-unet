# =========================================================================== #
#            Copyright © His Majesty the King in right of Ontario,            #
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
Top-level namespace for `landseg.execution.pipelines`.

Exposes selected public functions via lazy resolution to keep import
order simple and circular-free.
'''

# standard imports
from __future__ import annotations
import importlib
import typing

__all__ = [
    # types
    'Pipeline',
    'PreflightResult',
    'ProbeResult',
    'ProbeStatus',
    # functions
    'DataHarmonization',
    'DataIngestion',
    'DataPreparation',
    'ModelEvaluation',
    'ModelTraining',
    'WorldGridGeneration',
]


# for static check
if typing.TYPE_CHECKING:
    from .base import (
        Pipeline,
        PreflightResult,
        ProbeResult,
        ProbeStatus,
    )
    from .data_harmonize import (
        DataHarmonization,
    )
    from .data_ingest import (
        DataIngestion
    )
    from .data_prepare import (
        DataPreparation,
    )

    from .model_evaluate import (
        ModelEvaluation,
    )
    from .model_train import (
        ModelTraining,
    )

    from .world_grid import (
        WorldGridGeneration,
    )


def __getattr__(name: str):
    if name in {'Pipeline', 'PreflightResult', 'ProbeResult', 'ProbeStatus'}:
        obj = importlib.import_module('.base', __package__)
        return getattr(obj, name)

    if name in {'DataHarmonization'}:
        obj = importlib.import_module('.data_harmonize', __package__)
        return getattr(obj, name)

    if name in {'DataIngestion'}:
        obj = importlib.import_module('.data_ingest', __package__)
        return getattr(obj, name)

    if name in {'DataPreparation'}:
        obj = importlib.import_module('.data_prepare', __package__)
        return getattr(obj, name)

    if name in {'ModelEvaluation'}:
        obj = importlib.import_module('.model_evaluate', __package__)
        return getattr(obj, name)

    if name in {'ModelTraining'}:
        obj = importlib.import_module('.model_train', __package__)
        return getattr(obj, name)

    if name in {'WorldGridGeneration'}:
        obj = importlib.import_module('.world_grid', __package__)
        return getattr(obj, name)

    raise AttributeError(f'module {__name__!r} has no attribute {name!r}')
