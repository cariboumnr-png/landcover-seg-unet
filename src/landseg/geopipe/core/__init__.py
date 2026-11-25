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
    - `CategoricalSpec`: TypedDict for categorical raster specs.
    - `DataBlock`: Storage and interface for tiled geospatial blocks.
    - `DataBlockArrays`: Container for block-wise image and labels.
    - `DataBlockManifest`: TypedDict for block serialization manifest.
    - `DataCatalog`: Mapping container for block metadata.
    - `DataCatalogEntry`: TypedDict for individual block entry.
    - `DataSchema`: TypedDict for channel/band and label taxonomy.
    - `DomainPayload`: TypedDict for serialized domain tile map.
    - `DomainTile`: TypedDict for individual domain tile coordinates.
    - `DomainTileMap`: Mapping of valid spatial domain tiles.
    - `GridLayout`: Raster-agnostic grid layout of tile windows.
    - `GridPayload`: TypedDict for serialized grid payload.
    - `GridSpec`: Specification for constructing a world grid.
    - `ImageBandStats`: TypedDict for image band statistics.
    - `LabelScheme`: TypedDict for named reclassification scheme.
    - `PreparedSchema`: TypedDict for dataset preparation schema.
    - `RasterWindowDict`: TypedDict for raster window serialized dict.
    - `TargetHeadsSchema`: TypedDict for target heads hierarchy.
    - `name_xy`: Convert block name string to (x, y) coordinate tuple.
    - `xy_name`: Convert (x, y) coordinate tuple to block name string.
'''

# standard imports
import typing
# local imports
import landseg._lazy as _lazy

__all__ = [
    'CategoricalSpec',
    'DataBlock',
    'DataBlockArrays',
    'DataBlockManifest',
    'DataCatalog',
    'DataCatalogEntry',
    'DataSchema',
    'DomainPayload',
    'DomainTile',
    'DomainTileMap',
    'GridLayout',
    'GridPayload',
    'GridSpec',
    'ImageBandStats',
    'LabelScheme',
    'PreparedSchema',
    'RasterWindowDict',
    'TargetHeadsSchema',
    'name_xy',
    'xy_name',
]


# for static check
if typing.TYPE_CHECKING:
    from . import (
        data_schema,
        prepared_schema,
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
        DataCatalog,
        DataCatalogEntry,
    )
    from .data_schema import DataSchema
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
    from .label_scheme import (
        CategoricalSpec,
        LabelScheme,
    )
    from .prepared_schema import (
        ImageBandStats,
        PreparedSchema,
        TargetHeadsSchema,
    )


__getattr__, __dir__ = _lazy.attach(
    __name__,
    {
        '.coords': [
            'name_xy',
            'xy_name',
        ],
        '.data_block': [
            'DataBlock',
            'DataBlockArrays',
            'DataBlockManifest',
        ],
        '.data_catalog': [
            'DataCatalog',
            'DataCatalogEntry',
        ],
        '.data_schema': ['DataSchema'],
        '.domain_tile_map': [
            'DomainPayload',
            'DomainTile',
            'DomainTileMap',
        ],
        '.grid_layout': [
            'GridLayout',
            'GridPayload',
            'GridSpec',
            'RasterWindowDict',
        ],
        '.label_scheme': [
            'CategoricalSpec',
            'LabelScheme',
        ],
        '.prepared_schema': [
            'ImageBandStats',
            'PreparedSchema',
            'TargetHeadsSchema',
        ],
    },
    submodules=[
        'data_schema',
        'prepared_schema',
    ],
)
