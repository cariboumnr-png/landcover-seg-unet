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
Top-level namespace for `landseg.mlops`.

Provides standardized enterprise MLOps capabilities across CLI and notebooks:
Unity Catalog Model Registry integration, runtime environment detection,
lineage harvesting, and cloud volume storage resolution.

Public APIs:
    - `load_study_trials`: Load persisted sweep trials dataframe or summary.
    - `register_unity_catalog_model`: Register checkpoint to Unity Catalog.
    - `resolve_exp_root`: Resolve experiment root path across contexts.
    - `resolve_optuna_storage`: Resolve environment-safe Optuna URI.
    - `sync_optuna_study`: Persist Optuna study trials to volume storage.
    - `validate_output_root`: Validate user-designated session output root.
'''

# standard imports
import typing
# local imports
import landseg._lazy as _lazy

__all__ = [
    'load_study_trials',
    'register_unity_catalog_model',
    'resolve_exp_root',
    'resolve_optuna_storage',
    'sync_optuna_study',
    'validate_output_root',
]

# for static check
if typing.TYPE_CHECKING:
    from .registry import register_unity_catalog_model
    from .storage import (
        load_study_trials,
        resolve_exp_root,
        resolve_optuna_storage,
        sync_optuna_study,
        validate_output_root,
    )


__getattr__, __dir__ = _lazy.attach(__name__, {
    '.registry': ['register_unity_catalog_model'],
    '.storage': [
        'load_study_trials',
        'resolve_exp_root',
        'resolve_optuna_storage',
        'sync_optuna_study',
        'validate_output_root',
    ],
})
