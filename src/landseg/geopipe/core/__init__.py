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
    'DataBlock',
    'DomainTileMap',
    'GridLayout',
    'GridSpec',
    # functions
    # typing
    'DataBlockMeta',
    'DataCatalog',
    'DataSchema',
    'BlocksPartition',
    'CatalogEntry',
    'DomainMeta',
    'DomainPayload',
    'DomainTile',
    'GridPayload',
    'GridMeta',
    'ImageBandStats',
    'TransformSchema',
]

# for static check
if typing.TYPE_CHECKING:
    from .foundation_data_block import DataBlock, DataBlockMeta
    from .foundation_data_catalog import DataCatalog, CatalogEntry
    from .foundation_data_schema import DataSchema
    from .foundation_domain_map import DomainPayload, DomainMeta, DomainTile, DomainTileMap
    from .foundation_world_grid import GridSpec, GridPayload, GridMeta, GridLayout
    from .transform_types import BlocksPartition, ImageBandStats, TransformSchema

def __getattr__(name: str):

    if name in {'DataBlockMeta', 'DataBlock'}:
        return getattr(importlib.import_module('.foundation_data_block', __package__), name)

    if name in {'DataCatalog', 'CatalogEntry'}:
        return getattr(importlib.import_module('.foundation_data_catalog', __package__), name)

    if name in {'DataSchema'}:
        return getattr(importlib.import_module('.foundation_data_schema', __package__), name)

    if name in {'DomainPayload', 'DomainMeta', 'DomainTile', 'DomainTileMap'}:
        return getattr(importlib.import_module('.foundation_domain_map', __package__), name)

    if name in {'GridSpec', 'GridPayload', 'GridMeta', 'GridLayout'}:
        return getattr(importlib.import_module('.foundation_world_grid', __package__), name)

    if name in {'BlocksPartition', 'ImageBandStats', 'TransformSchema'}:
        return getattr(importlib.import_module('.transform_types', __package__), name)

    raise AttributeError(f'module {__name__!r} has no attribute {name!r}')
