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
Diagnostic probe library for pre-flight readiness checks.

Exposes specialized probes for hardware, filesystem storage, spatial
contracts, run ledgers, model specifications, and optimization studies.
'''

# standard imports
from __future__ import annotations
import importlib
import typing

__all__ = [
    'canonical_pool_state',
    'checkpoint_ready',
    'collision_policy',
    'crs_info',
    'dataset_targets',
    'dir_writable',
    'eval_split',
    'grid_extent',
    'grid_origin',
    'grid_specs',
    'hardware_info',
    'model_body',
    'past_runs',
    'pending_batches',
    'pixel_size',
    'spatial_reference',
    'split_ratios',
    'target_file_exists',
]


# for static check
if typing.TYPE_CHECKING:
    from .dataset import (
        dataset_targets,
        split_ratios,
    )
    from .filesystem import (
        dir_writable,
        target_file_exists,
    )
    from .hardware import (
        hardware_info,
    )
    from .ledger import (
        canonical_pool_state,
        past_runs,
        pending_batches,
    )
    from .model import (
        checkpoint_ready,
        eval_split,
        model_body,
    )
    from .policy import (
        collision_policy,
    )
    from .spatial import (
        crs_info,
        grid_extent,
        grid_origin,
        grid_specs,
        pixel_size,
        spatial_reference,
    )


def __getattr__(name: str):
    if name in {
        'dataset_targets',
        'split_ratios',
    }:
        obj = importlib.import_module('.dataset', __package__)
        return getattr(obj, name)

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
        'canonical_pool_state',
        'past_runs',
        'pending_batches',
        'probe_ledger',
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
        'crs_info',
        'grid_extent',
        'grid_origin',
        'grid_specs',
        'pixel_size',
        'spatial_reference',
    }:
        obj = importlib.import_module('.spatial', __package__)
        return getattr(obj, name)

    raise AttributeError(f'module {__name__!r} has no attribute {name!r}')


