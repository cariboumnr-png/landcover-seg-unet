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
    'DataBlockInputs',
    'DataBlockConfig',
    'DomainTileMap',
    'GridLayout',
    'GridSpec',
    # functions
    # typing
    'CategoricalSpecs',
    'DataBlockManifest',
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
    'LabelScheme',
    'LabelSchemes',
    'TaxonomySpecs',
    'TransformSchema',
    'PartitionSummary',
]

# for static check
if typing.TYPE_CHECKING:
    from .categorical_types import (
        CategoricalSpecs,
        LabelScheme,
        LabelSchemes,
        TaxonomySpecs,
    )
    from .data_block import (
        DataBlock,
        DataBlockConfig,
        DataBlockInputs,
        DataBlockManifest,
    )
    from .data_catalog import DataCatalog, CatalogEntry
    from .data_schema import DataSchema
    from .domain_tilemap import (
        DomainPayload,
        DomainMeta,
        DomainTile,
        DomainTileMap
    )
    from .grid_layout import (
        GridSpec,
        GridPayload,
        GridMeta,
        GridLayout
    )
    from .prepared_dateset_types import (
        BlocksPartition,
        ImageBandStats,
        TransformSchema,
        PartitionSummary
    )


def __getattr__(name: str):
    if name in {
        'CategoricalSpecs',
        'LabelScheme',
        'LabelSchemes',
        'TaxonomySpecs',
    }:
        obj = importlib.import_module('.categorical_types', __package__)
        return getattr(obj, name)

    if name in {
        'GridSpec',
        'GridPayload',
        'GridMeta',
        'GridLayout'
    }:
        obj = importlib.import_module('.grid_layout', __package__)
        return getattr(obj, name)

    if name in {
        'DataBlock',
        'DataBlockConfig',
        'DataBlockInputs',
        'DataBlockManifest',
    }:
        obj = importlib.import_module('.data_block', __package__)
        return getattr(obj, name)

    if name in {
        'DataCatalog',
        'CatalogEntry'
    }:
        obj = importlib.import_module('.data_catalog', __package__)
        return getattr(obj, name)

    if name in {'DataSchema'}:
        obj = importlib.import_module('.data_schema', __package__)
        return getattr(obj, name)

    if name in {
        'DomainPayload',
        'DomainMeta',
        'DomainTile',
        'DomainTileMap'
    }:
        obj = importlib.import_module('.domain_tilemap', __package__)
        return getattr(obj, name)

    if name in {
        'BlocksPartition',
        'ImageBandStats',
        'TransformSchema',
        'PartitionSummary'
    }:
        obj = importlib.import_module('.prepare_blocks_types', __package__)
        return getattr(obj, name)

    raise AttributeError(f'module {__name__!r} has no attribute {name!r}')
