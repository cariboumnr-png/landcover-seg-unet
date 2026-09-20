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
    - `DatasetCatalog`: Mapping container for block metadata.
    - `DomainTileMap`: Mapping of valid spatial domain tiles.
    - `GridLayout`: Raster-agnostic grid layout of tile windows.
    - `GridSpec`: Specification for constructing a world grid.
    - `get_grid_report_fpath`: Canonical path to grid report artifact.
    - `load_grid_from_fpath`: Load world grid layout directly from file.
    - `read_grid_report`: Read grid report JSON and extract summary.
    - `CategoricalSpec`: TypedDict for categorical raster specs.
    - `DataBlockManifest`: TypedDict for block serialization manifest.
    - `DatasetBlockMeta`: TypedDict for individual block entry.
    - `DatasetSchema`: TypedDict for channel/band and label taxonomy.
    - `DomainMeta`: TypedDict for domain tile map metadata.
    - `DomainPayload`: TypedDict for serialized domain tile map.
    - `DomainTile`: TypedDict for individual domain tile coordinates.
    - `GridMeta`: TypedDict for grid metadata.
    - `GridPayload`: TypedDict for serialized grid payload.
    - `LabelScheme`: TypedDict for named reclassification scheme.
    - `RasterReader`: Type alias for raster reader protocol.
    - `RasterWindow`: Type alias for raster window definition.
    - `RasterWindowDict`: TypedDict for raster window serialized dict.
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
    'DatasetCatalog',
    'DomainTileMap',
    'GridLayout',
    'GridSpec',
    # functions
    'get_grid_report_fpath',
    'load_grid_from_fpath',
    'read_grid_report',
    # typing
    'CategoricalSpec',
    'DataBlockManifest',
    'DatasetBlockMeta',
    'DatasetSchema',
    'DomainMeta',
    'DomainPayload',
    'DomainTile',
    'GridMeta',
    'GridPayload',
    'LabelScheme',
    'RasterReader',
    'RasterWindow',
    'RasterWindowDict',
    'TaxonomySpec',
]


# for static check
if typing.TYPE_CHECKING:
    from .categorical import (
        CategoricalSpec,
        LabelScheme,
        TaxonomySpec,
    )
    from .data_block import (
        DataBlock,
        DataBlockArrays,
        DataBlockManifest,
    )
    from .dataset_catalog import (
        DatasetBlockMeta,
        DatasetCatalog,
    )
    from .dataset_schema import (
        DatasetSchema,
    )
    from .domain_tile_map import (
        DomainMeta,
        DomainPayload,
        DomainTile,
        DomainTileMap,
    )
    from .grid_layout import (
        GridLayout,
        GridMeta,
        GridPayload,
        GridSpec,
        RasterReader,
        RasterWindow,
        RasterWindowDict,
        get_grid_report_fpath,
        load_grid_from_fpath,
        read_grid_report,
    )


def __getattr__(name: str):
    if name in {
        'CategoricalSpec',
        'LabelScheme',
        'TaxonomySpec',
    }:
        obj = importlib.import_module('.categorical', __package__)
        return getattr(obj, name)

    if name in {
        'DataBlock',
        'DataBlockArrays',
        'DataBlockManifest',
    }:
        obj = importlib.import_module('.data_block', __package__)
        return getattr(obj, name)

    if name in {
        'DatasetBlockMeta',
        'DatasetCatalog',
    }:
        obj = importlib.import_module('.dataset_catalog', __package__)
        return getattr(obj, name)

    if name in {
        'DatasetSchema',
        'dataset_schema',
    }:
        obj = importlib.import_module('.dataset_schema', __package__)
        return obj if name == 'dataset_schema' else getattr(obj, name)

    if name in {
        'DomainMeta',
        'DomainPayload',
        'DomainTile',
        'DomainTileMap',
    }:
        obj = importlib.import_module('.domain_tile_map', __package__)
        return getattr(obj, name)

    if name in {
        'GridLayout',
        'GridMeta',
        'GridPayload',
        'GridSpec',
        'RasterReader',
        'RasterWindow',
        'RasterWindowDict',
        'get_grid_report_fpath',
        'load_grid_from_fpath',
        'read_grid_report',
    }:
        obj = importlib.import_module('.grid_layout', __package__)
        return getattr(obj, name)

    raise AttributeError(f'module {__name__!r} has no attribute {name!r}')
