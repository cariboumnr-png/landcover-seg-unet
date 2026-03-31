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
Top-level namespace for `landseg.geopipe.core`.

Exposes selected public functions via lazy resolution to keep import
order simple and circular-free.
'''

from __future__ import annotations
import importlib
import typing

__all__ = [
    # classes
    'BlockMeta',
    'CatalogEntry',
    'DataBlock',
    'DomainTileMap',
    'GridLayout',
    'GridSpec',
    # functions
    'name_xy',
    'xy_name',
    'open_rasters',
    # typing
    'BlocksCatalog',
    'CatalogMeta',
    'BlockSplitPaths',
    'ImageBandStats',
    'TransformSchema',
    'GridLayoutPayload',
]

# for static check
if typing.TYPE_CHECKING:
    from .foundation_catalog import BlocksCatalog, CatalogMeta, CatalogEntry
    from .foundation_data_block import BlockMeta, DataBlock
    from .foundation_domain_map import DomainTileMap
    from .foundation_world_grid import GridLayout, GridLayoutPayload, GridSpec
    from .transform_types import BlockSplitPaths, ImageBandStats, TransformSchema
    from .utils import name_xy, xy_name, open_rasters

def __getattr__(name: str):

    if name in ['BlockMeta', 'DataBlock']:
        return getattr(importlib.import_module('.foundation_data_block', __package__), name)

    if name in ['BlocksCatalog', 'CatalogMeta', 'CatalogEntry']:
        return getattr(importlib.import_module('.foundation_catalog', __package__), name)

    if name in ['DomainTileMap']:
        return getattr(importlib.import_module('.foundation_domain_map', __package__), name)

    if name in ['GridLayout', 'GridLayoutPayload', 'GridSpec']:
        return getattr(importlib.import_module('.foundation_world_grid', __package__), name)

    if name in ['BlockSplitPaths', 'ImageBandStats', 'TransformSchema']:
        return getattr(importlib.import_module('.transform_types', __package__), name)

    if name in ['name_xy', 'xy_name', 'open_rasters']:
        return getattr(importlib.import_module('.utils', __package__), name)

    raise AttributeError(name)
