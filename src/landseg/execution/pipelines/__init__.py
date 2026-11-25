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

'''
Top-level namespace for `landseg.execution.pipelines`.

Exposes selected public functions via lazy resolution to keep import
order simple and circular-free.
'''

# standard imports
import typing
# local imports
import landseg._lazy as _lazy

__all__ = [
    'BasePipeline',
    'DataHarmonization',
    'DataIngestion',
    'DataPreparation',
    'GeoPipeline',
    'ModelEvaluation',
    'ModelTraining',
    'SessionPipeline',
    'WorldGridGeneration',
]


# for static check
if typing.TYPE_CHECKING:
    from .base import (
        BasePipeline,
        GeoPipeline,
        SessionPipeline,
    )
    from .data_harmonize import DataHarmonization
    from .data_ingest import DataIngestion
    from .data_prepare import DataPreparation
    from .model_evaluate import ModelEvaluation
    from .model_train import ModelTraining
    from .world_grid import WorldGridGeneration


__getattr__, __dir__ = _lazy.attach(__name__, {
    '.base': [
        'BasePipeline',
        'GeoPipeline',
        'SessionPipeline',
    ],
    '.data_harmonize': ['DataHarmonization'],
    '.data_ingest': ['DataIngestion'],
    '.data_prepare': ['DataPreparation'],
    '.model_evaluate': ['ModelEvaluation'],
    '.model_train': ['ModelTraining'],
    '.world_grid': ['WorldGridGeneration'],
})
