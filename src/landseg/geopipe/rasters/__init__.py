# =========================================================================== #
#           Copyright © His Majesty the King in right of Ontario,           #
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
Top-level namespace for `landseg.geopipe.rasters`.

Exposes raster warping, stacking, masking, and metadata manipulation APIs.

Public APIs:
    - `add_band_description_to_vrt`: Add band description labels to VRT.
    - `add_tag_to_vrt`: Add metadata tags to a VRT raster file.
    - `build_vrt_mosaic`: Merge multiple raster tiles into a mosaic VRT.
    - `resolve_resampling_algorithm`: Resolve rasterio resampling enum.
    - `stack_rasters`: Stack spatially aligned rasters into a VRT.
    - `unify_nodata_mask`: Create a 1-band valid pixel mask across bands.
    - `write_warped_vrt`: Write a virtual warped raster (VRT) to disk.
'''

# standard imports
from __future__ import annotations
import importlib
import typing

__all__ = [
    # types
    'Affine',
    'BoundingBox',
    'DatasetReader',
    'Window',
    # classes
    'DatasetInfo',
    # functions
    'get_info',
    'read_raster',
    'read_rasters',
    'read_window',
    'add_band_description_to_vrt',
    'add_tag_to_vrt',
    'build_vrt_mosaic',
    'resolve_resampling_algorithm',
    'stack_rasters',
    'unify_nodata_mask',
    'write_warped_vrt',
]


# for static check
if typing.TYPE_CHECKING:
    from ._types import (
        Affine,
        BoundingBox,
        DatasetReader,
        Window,
    )
    from .io import (
        DatasetInfo,
        get_info,
        read_raster,
        read_rasters,
        read_window,
    )
    from .mask import (
        unify_nodata_mask,
    )
    from .mosaic import (
        build_vrt_mosaic
    )
    from .stack import (
        stack_rasters
    )
    from .vrt import(
        add_tag_to_vrt,
        add_band_description_to_vrt,
    )
    from .warp import (
        resolve_resampling_algorithm,
        write_warped_vrt,
    )


def __getattr__(name: str):
    if name in {
        'Affine',
        'BoundingBox',
        'DatasetReader',
        'Window',
    }:
        obj = importlib.import_module('._types', __package__)
        return getattr(obj, name)

    if name in {
        'DatasetInfo',
        'get_info',
        'read_raster',
        'read_rasters',
        'read_window',
    }:
        obj = importlib.import_module('.io', __package__)
        return getattr(obj, name)

    if name in {
        'unify_nodata_mask'
    }:
        obj = importlib.import_module('.mask', __package__)
        return getattr(obj, name)

    if name in {
        'build_vrt_mosaic'
    }:
        obj = importlib.import_module('.mosaic', __package__)
        return getattr(obj, name)

    if name in {
        'stack_rasters'
    }:
        obj = importlib.import_module('.stack', __package__)
        return getattr(obj, name)

    if name in {
        'add_tag_to_vrt',
        'add_band_description_to_vrt',
    }:
        obj = importlib.import_module('.vrt', __package__)
        return getattr(obj, name)

    if name in {
        'resolve_resampling_algorithm',
        'write_warped_vrt',
    }:
        obj = importlib.import_module('.warp', __package__)
        return getattr(obj, name)


    raise AttributeError(f'module {__name__!r} has no attribute {name!r}')
