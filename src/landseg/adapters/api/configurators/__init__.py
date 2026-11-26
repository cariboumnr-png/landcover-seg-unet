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
Top-level namespace for `landseg.adapters.api.configurators`.

Exposes selected configurator classes via lazy resolution to keep import
order simple and circular-free.

Public APIs:
    - `BaseConfigurator`: Base configurator for execution commands.
    - `DataHarmonizationConfigurator`: Configurator for harmonization.
    - `DataIngestionConfigurator`: Configurator for data ingestion.
    - `DataPreparationConfigurator`: Configurator for data preparation.
    - `StudySweepConfigurator`: Configurator for study sweeps.
    - `TrainingSessionConfigurator`: Configurator for training sessions.
    - `WorldGridConfigurator`: Configurator for world grid generation.
'''

# standard imports
import typing
# local imports
import landseg._lazy as _lazy

__all__ = [
    'BaseConfigurator',
    'DataHarmonizationConfigurator',
    'DataIngestionConfigurator',
    'DataPreparationConfigurator',
    'StudySweepConfigurator',
    'TrainingSessionConfigurator',
    'WorldGridConfigurator',
]


# for static check
if typing.TYPE_CHECKING:
    from .base import BaseConfigurator
    from .data_harmonize import DataHarmonizationConfigurator
    from .data_ingest import DataIngestionConfigurator
    from .data_prepare import DataPreparationConfigurator
    from .model_train import TrainingSessionConfigurator
    from .study_sweep import StudySweepConfigurator
    from .world_grid import WorldGridConfigurator


__getattr__, __dir__ = _lazy.attach(__name__, {
    '.base': ['BaseConfigurator'],
    '.data_harmonize': ['DataHarmonizationConfigurator'],
    '.data_ingest': ['DataIngestionConfigurator'],
    '.data_prepare': ['DataPreparationConfigurator'],
    '.model_train': ['TrainingSessionConfigurator'],
    '.study_sweep': ['StudySweepConfigurator'],
    '.world_grid': ['WorldGridConfigurator'],
})
