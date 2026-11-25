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
Top-level namespace for `landseg.geopipe`.

Coordinates raw geospatial raster harmonization, domain tile mapping,
data block construction, and runtime dataset specification creation
through structured pipelines and lazy module resolution.

Public APIs:
    - `build_dataspec`: Assembles runtime DataSpecs from artifacts.
    - `validate_feature_specs`: Validate requested feature specs.
    - `validate_grid_reference`: Validate a persisted world grid reference.
    - `validate_grid_specs`: Validate world grid definition inputs.
    - `validate_manifest_contract`: Validate input dataset manifest entries.
    - `validate_pool_compatibility`: Validate grid/pool compatibility.
    - `validate_prepare_semantics`: Validate preparation feature & head specs.
    - `validate_raster_alignment`: Validate raster/grid alignment for ingest.
'''

# standard imports
from __future__ import annotations
import importlib
import typing

__all__ = [
    # functions
    'build_dataspec',
    'validate_feature_specs',
    'validate_grid_reference',
    'validate_grid_specs',
    'validate_manifest_contract',
    'validate_pool_compatibility',
    'validate_prepare_semantics',
    'validate_raster_alignment',
]


# for static check
if typing.TYPE_CHECKING:
    from .factory import (
        build_dataspec,
    )
    from .validation import (
        validate_feature_specs,
        validate_grid_reference,
        validate_grid_specs,
        validate_manifest_contract,
        validate_pool_compatibility,
        validate_prepare_semantics,
        validate_raster_alignment,
    )


def __getattr__(name: str):
    if name in {'build_dataspec'}:
        obj = importlib.import_module('.factory', __package__)
        return getattr(obj, name)

    if name in {
        'validate_feature_specs',
        'validate_grid_reference',
        'validate_grid_specs',
        'validate_manifest_contract',
        'validate_pool_compatibility',
        'validate_prepare_semantics',
        'validate_raster_alignment',
    }:
        obj = importlib.import_module('.validation', __package__)
        return getattr(obj, name)

    raise AttributeError(f'module {__name__!r} has no attribute {name!r}')
