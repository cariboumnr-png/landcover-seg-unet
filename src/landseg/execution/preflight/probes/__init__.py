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
    'hardware_info',
    'spatial',

    'past_runs',

    'model_body',

    'dir_writable',
    'file_exists',

    'crs_info',
    'grid_extent',
    'grid_origin',
    'grid_specs',
    'pixel_size',
    'spatial_reference',
]


# for static check
if typing.TYPE_CHECKING:
    from .hardware import (
        hardware_info,
    )
    from .ledger import (
        past_runs,
    )
    from .model import (
        model_body,
    )
    from .spatial import (
        crs_info,
        grid_extent,
        grid_origin,
        grid_specs,
        pixel_size,
        spatial_reference,
    )
    from .filesystem import (
        dir_writable,
        file_exists,
    )

def __getattr__(name: str):
    if name in {'probe_hardware'}:
        obj = importlib.import_module('.hardware', __package__)
        return getattr(obj, name)

    if name in {
        'probe_ledger',
        'past_runs'
    }:
        obj = importlib.import_module('.ledger', __package__)
        return getattr(obj, name)

    if name in {
        'model_body'
    }:
        obj = importlib.import_module('.model', __package__)
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

    if name in {
        'dir_writable',
        'file_exists',
    }:
        obj = importlib.import_module('.filesystem', __package__)
        return getattr(obj, name)

    raise AttributeError(f'module {__name__!r} has no attribute {name!r}')
