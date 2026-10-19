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
Top-level namespace for `landseg.geopipe.harmonize.rasters`.

Exposes raster warping, stacking, masking, and metadata manipulation APIs.

Public APIs:
    - `ProcessedRasters`: Container for processed raster paths dictionaries.
    - `build_vrt_mosaic`: Merge multiple raster tiles into a mosaic VRT.
    - `raster_processor`: Harmonize compiled raster sources onto grid.
    - `unify_nodata_mask`: Create a 1-band valid pixel mask across bands.
'''

# standard imports
from __future__ import annotations
import importlib
import typing

__all__ = [
    # classes
    'ProcessedRasters',
    # functions
    'build_vrt_mosaic',
    'raster_processor',
    'unify_nodata_mask',
]


# for static check
if typing.TYPE_CHECKING:
    from .mask import (
        unify_nodata_mask,
    )
    from .mosaic import (
        build_vrt_mosaic
    )
    from .processor import (
        ProcessedRasters,
        raster_processor,
    )


def __getattr__(name: str):
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
        'ProcessedRasters',
        'raster_processor',
    }:
        obj = importlib.import_module('.processor', __package__)
        return getattr(obj, name)

    raise AttributeError(f'module {__name__!r} has no attribute {name!r}')
