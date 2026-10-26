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
    - `DataBlockArrays`: Container for block-wise image and labels.
    - `DataCatalog`: Mapping container for block metadata.
    - `DomainTileMap`: Mapping of valid spatial domain tiles.
    - `GridLayout`: Raster-agnostic grid layout of tile windows.
    - `GridSpec`: Specification for constructing a world grid.
    - `name_xy`: Convert block name string to (x, y) coordinate tuple.
    - `xy_name`: Convert (x, y) coordinate tuple to block name string.
    - `CategoricalSpec`: TypedDict for categorical raster specs.
    - `DataBlockManifest`: TypedDict for block serialization manifest.
    - `DataCatalogEntry`: TypedDict for individual block entry.
    - `DataSchema`: TypedDict for channel/band and label taxonomy.
    - `DomainPayload`: TypedDict for serialized domain tile map.
    - `DomainTile`: TypedDict for individual domain tile coordinates.
    - `GridPayload`: TypedDict for serialized grid payload.
    - `ImageBandStats`: TypedDict for image band statistics.
    - `LabelScheme`: TypedDict for named reclassification scheme.
    - `PreparedSchema`: TypedDict for dataset preparation schema.
    - `RasterReader`: Type alias for raster reader protocol.
    - `RasterWindow`: Type alias for raster window definition.
    - `RasterWindowDict`: TypedDict for raster window serialized dict.
    - `TargetHeadsSchema`: TypedDict for target heads hierarchy.
    - `TaxonomySpec`: TypedDict for domain taxonomy specification.
'''

# standard imports
from __future__ import annotations
import importlib
import typing

__all__ = [
    # classes
    'DataBlock',
    'DataBlockArrays',
    'DataCatalog',
    'DomainTileMap',
    'GridLayout',
    'GridSpec',
    # functions
    'name_xy',
    'xy_name',
    # typing
    'CategoricalSpec',
    'DataBlockManifest',
    'DataCatalogEntry',
    'DataSchema',
    'DomainPayload',
    'DomainTile',
    'GridPayload',
    'ImageBandStats',
    'LabelScheme',
    'PreparedSchema',
    'RasterWindowDict',
    'TargetHeadsSchema',
    'TaxonomySpec',
]


# for static check
if typing.TYPE_CHECKING:
    from .label_scheme import (
        CategoricalSpec,
        LabelScheme,
        TaxonomySpec,
    )
    from .coords import (
        name_xy,
        xy_name,
    )
    from .data_block import (
        DataBlock,
        DataBlockArrays,
        DataBlockManifest,
    )
    from .data_catalog import (
        DataCatalogEntry,
        DataCatalog,
    )
    from .data_schema import (
        DataSchema,
    )
    from .domain_tile_map import (
        DomainPayload,
        DomainTile,
        DomainTileMap,
    )
    from .grid_layout import (
        GridLayout,
        GridPayload,
        GridSpec,
        RasterWindowDict,
    )
    from .prepared_schema import (
        ImageBandStats,
        PreparedSchema,
        TargetHeadsSchema,
    )


def __getattr__(name: str):
    if name in {
        'CategoricalSpec',
        'LabelScheme',
        'TaxonomySpec',
    }:
        obj = importlib.import_module('.label_scheme', __package__)
        return getattr(obj, name)

    if name in {
        'name_xy',
        'xy_name',
    }:
        obj = importlib.import_module('.coords', __package__)
        return getattr(obj, name)

    if name in {
        'DataBlock',
        'DataBlockArrays',
        'DataBlockManifest',
    }:
        obj = importlib.import_module('.data_block', __package__)
        return getattr(obj, name)

    if name in {
        'DataCatalogEntry',
        'DataCatalog',
    }:
        obj = importlib.import_module('.data_catalog', __package__)
        return getattr(obj, name)

    if name in {
        'DataSchema',
    }:
        obj = importlib.import_module('.data_schema', __package__)
        return getattr(obj, name)

    if name in {
        'DomainPayload',
        'DomainTile',
        'DomainTileMap',
    }:
        obj = importlib.import_module('.domain_tile_map', __package__)
        return getattr(obj, name)

    if name in {
        'GridLayout',
        'GridPayload',
        'GridSpec',
        'RasterWindowDict',
    }:
        obj = importlib.import_module('.grid_layout', __package__)
        return getattr(obj, name)

    if name in {
        'ImageBandStats',
        'PreparedSchema',
        'TargetHeadsSchema',
    }:
        obj = importlib.import_module('.prepared_schema', __package__)
        return getattr(obj, name)

    if name in {
        'data_schema',
        'prepared_schema',
    }:
        return importlib.import_module(f'.{name}', __package__)

    raise AttributeError(f'module {__name__!r} has no attribute {name!r}')
