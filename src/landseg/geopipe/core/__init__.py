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

Exposes selected core abstractions and contracts via lazy resolution to
keep import order simple and circular-free.

Public APIs:
    - `DataBlock`: Storage and interface for tiled geospatial blocks.
    - `DataBlockInputs`: Named container for block input arrays.
    - `DataBlockConfig`: Configuration for block layout and storage.
    - `DomainTileMap`: Mapping of valid spatial domain tiles.
    - `GridLayout`: Raster-agnostic grid layout of tile windows.
    - `GridSpec`: Specification for constructing a world grid.
    - `CategoricalSpecs`: TypedDict for categorical raster specs.
    - `DataBlockManifest`: TypedDict for block serialization manifest.
    - `DataCatalog`: TypedDict for dataset-wide catalog indexing.
    - `DataSchema`: TypedDict for channel/band and label taxonomy.
    - `BlocksPartition`: TypedDict mapping block IDs across splits.
    - `CatalogEntry`: TypedDict for an individual block entry.
    - `DomainMeta`: TypedDict for domain tile map metadata.
    - `DomainPayload`: TypedDict for serialized domain tile map.
    - `DomainTile`: TypedDict for individual domain tile coordinates.
    - `GridPayload`: TypedDict for serialized grid payload.
    - `GridMeta`: TypedDict for grid metadata.
    - `ImageBandStats`: TypedDict for image band statistics.
    - `LabelScheme`: TypedDict for named reclassification scheme.
    - `LabelSchemes`: Type alias for mapping names to `LabelScheme`.
    - `TaxonomySpecs`: TypedDict for domain taxonomy specification.
    - `TransformSchema`: TypedDict for dataset-wide transform schema.
    - `PartitionSummary`: TypedDict for split and hydration summary.
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
    'PreparedSchema',
    'TargetHeadsSchema',
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
        PreparedSchema,
        TargetHeadsSchema,
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
        'PreparedSchema',
        'TargetHeadsSchema',
        'PartitionSummary'
    }:
        obj = importlib.import_module('.prepared_dateset_types', __package__)
        return getattr(obj, name)

    raise AttributeError(f'module {__name__!r} has no attribute {name!r}')
