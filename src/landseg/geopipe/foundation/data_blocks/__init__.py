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
Top-level namespace for `landseg.geopipe.foundation.data_blocks`.

Exposes selected public functions via lazy resolution to keep import
order simple and circular-free.
'''

from __future__ import annotations
import importlib
import typing

__all__ = [
    # classes
    'BlockBuilder',
    'BlockBuildingConfig',
    'CatalogUpdateContext',
    'MappingConfig',
    # functions
    'map_rasters',
    'run_blocks_building',
    'validate_geometry',
    'update_catalog',
    'update_meta',
    # typing
    'BuilderConfig',
    'GeometrySummary',
]

# for static check
if typing.TYPE_CHECKING:
    from .pipeline import BlockBuildingConfig, run_blocks_building
    from .builder import BlockBuilder, BuilderConfig
    from .catalogue import CatalogUpdateContext, update_catalog, update_meta
    from .geometry import GeometrySummary, validate_geometry
    from .mapper import MappingConfig, map_rasters

def __getattr__(name: str):

    if name in ['BlockBuilder', 'BuilderConfig']:
        return getattr(importlib.import_module('.builder', __package__), name)
    if name in ['CatalogUpdateContext', 'update_catalog', 'update_meta']:
        return getattr(importlib.import_module('.catalogue', __package__), name)
    if name in ['GeometrySummary', 'validate_geometry']:
        return getattr(importlib.import_module('.geometry', __package__), name)
    if name in ['MappingConfig', 'map_rasters']:
        return getattr(importlib.import_module('.mapper', __package__), name)
    if name in ['BlockBuildingConfig', 'run_blocks_building']:
        return getattr(importlib.import_module('.pipeline', __package__), name)

    raise AttributeError(name)
