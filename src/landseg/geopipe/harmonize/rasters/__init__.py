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
    - `warp_to_grid`: Reproject and snap input raster to grid as a VRT.
    - `stack_rasters`: Stack feature and label rasters into composite VRTs.
    - `unify_nodata_mask`: Create a 1-band valid pixel mask across bands.
    - `add_band_description_to_vrt`: Add band descriptions to a VRT file.
    - `add_tag_to_vrt`: Attach metadata tags to a VRT raster file.
'''

# standard imports
from __future__ import annotations
import importlib
import typing

__all__ = [
    # classes
    # functions
    'warp_to_grid',
    'stack_rasters',
    'unify_nodata_mask',
    'add_band_description_to_vrt',
    'add_tag_to_vrt',
]

# for static check
if typing.TYPE_CHECKING:
    from .spatial import (
        warp_to_grid,
    )

    from .metadata import (
        add_band_description_to_vrt,
        add_tag_to_vrt,
    )

    from .mask import (
        unify_nodata_mask,
    )

    from .stack import(
        stack_rasters,
    )


def __getattr__(name: str):

    if name in {
        'warp_to_grid',
    }:
        return getattr(importlib.import_module('.spatial', __package__), name)

    if name in {
        'add_band_description_to_vrt',
        'add_tag_to_vrt',
    }:
        return getattr(
            importlib.import_module('.metadata', __package__), name
        )

    if name in {
        'unify_nodata_mask',
    }:
        return getattr(
            importlib.import_module('.mask', __package__), name
        )

    if name in {
        'stack_rasters',
    }:
        return getattr(
            importlib.import_module('.stack', __package__), name
        )

    raise AttributeError(f'module {__name__!r} has no attribute {name!r}')
