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
Preflight validation probes package.

Provides automated environment, hardware, dataset, policy, and spatial
probes that can be combined into custom pipelines.
'''

# standard imports
from __future__ import annotations
import importlib
import typing

__all__ = [
    'checkpoint_ready',
    'collision_policy',
    'dir_writable',
    'eval_split',
    'grid_inputs',
    'grid_reference',
    'hardware_info',
    'ingestion_pool_state',
    'manifest_contract',
    'model_body',
    'past_runs',
    'pending_harmonization',
    'pending_ingestion',
    'prepared_blocks_state',
    'target_file_exists',
]


# for static check
if typing.TYPE_CHECKING:
    from .filesystem import (
        dir_writable,
        target_file_exists,
    )
    from .hardware import (
        hardware_info,
    )
    from .ledger import (
        ingestion_pool_state,
        past_runs,
        pending_harmonization,
        pending_ingestion,
        prepared_blocks_state,
    )
    from .model import (
        checkpoint_ready,
        eval_split,
        model_body,
    )
    from .policy import (
        collision_policy,
    )
    from .stage_contracts import (
        grid_inputs,
        grid_reference,
        manifest_contract,
    )


def __getattr__(name: str):

    if name in {
        'dir_writable',
        'target_file_exists',
    }:
        obj = importlib.import_module('.filesystem', __package__)
        return getattr(obj, name)

    if name in {
        'hardware_info',
    }:
        obj = importlib.import_module('.hardware', __package__)
        return getattr(obj, name)

    if name in {
        'ingestion_pool_state',
        'past_runs',
        'pending_harmonization',
        'pending_ingestion',
        'prepared_blocks_state',
    }:
        obj = importlib.import_module('.ledger', __package__)
        return getattr(obj, name)

    if name in {
        'checkpoint_ready',
        'eval_split',
        'model_body',
    }:
        obj = importlib.import_module('.model', __package__)
        return getattr(obj, name)

    if name in {
        'collision_policy',
    }:
        obj = importlib.import_module('.policy', __package__)
        return getattr(obj, name)

    if name in {
        'grid_inputs',
        'grid_reference',
        'manifest_contract',
    }:
        obj = importlib.import_module('.stage_contracts', __package__)
        return getattr(obj, name)

    raise AttributeError(f'module {__name__!r} has no attribute {name!r}')
