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
VRT raster metadata and band description utilities.

This module provides helper functions to attach band descriptions and
arbitrary key-value metadata tags to GDAL Virtual Raster (VRT) files.

Public APIs:
    - `add_band_description_to_vrt`: Add band descriptions to a VRT file.
    - `add_tag_to_vrt`: Attach metadata tags to a VRT raster file.
'''

# standard imports
from __future__ import annotations
import typing
# third-party imports
import rasterio


# ----- public functions
def add_band_description_to_vrt(
    vrt_fpath: str,
    band_mapping: dict[int, str],
) -> None:
    '''
    Add band description labels to a VRT raster file.

    Args:
        vrt_fpath:
            File path to the VRT dataset.
        band_mapping:
            Mapping from 1-based band index to band description string.
    '''
    with rasterio.open(vrt_fpath, 'r+') as vrt:
        if len(band_mapping) != vrt.count:
            raise ValueError(
                f'Expected {vrt.count} band descriptions, '
                f'got {len(band_mapping)}'
            )
        for band, name in band_mapping.items():
            vrt.set_band_description(int(band), name)


def add_tag_to_vrt(
    vrt_fpath: str,
    **kwargs: typing.Any,
) -> None:
    '''
    Add metadata tags to a VRT raster file.

    Args:
        vrt_fpath:
            File path to the VRT dataset.
        **kwargs:
            Key-value pairs to store as metadata tags.
    '''
    tags = {k: v for k, v in kwargs.items() if v is not None}
    if not tags:
        return
    with rasterio.open(vrt_fpath, 'r+') as vrt:
        vrt.update_tags(**tags)
