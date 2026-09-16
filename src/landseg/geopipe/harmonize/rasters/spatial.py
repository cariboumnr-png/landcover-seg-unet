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
Spatial grid specification and raster warping operations.

This module provides functions to reproject, resample, and snap input
rasters to a canonical world grid layout as GDAL Virtual Rasters (VRT).

Public APIs:
    - `warp_to_grid`: Reproject and snap input raster to grid as a VRT.
'''

# standard imports
from __future__ import annotations
import os
# third-party imports
import rasterio
import rasterio.enums
import rasterio.shutil
import rasterio.vrt
# local imports
import landseg.geopipe.core as geo_core


# ----- public functions
def warp_to_grid(
    *,
    input_path: str,
    output_path: str,
    world_grid: geo_core.GridLayout,
    is_categorical: bool = False,
    resampling_method: str | None = None,
) -> str:
    '''
    Reproject and snap an input raster to target grid as a VRT.

    Args:
        input_path:
            Path to the raw source GeoTIFF.
        output_path:
            Destination path for the harmonized Virtual Raster (.vrt).
        world_grid:
            A `GridLayout` instance.
        is_categorical:
            If True, strictly uses nearest-neighbor resampling.
        resampling_method:
            Optional string override ('nearest', 'bilinear', 'cubic').

    Returns:
        str:
            Absolute path to output harmonized Virtual Raster file.
    '''
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    if is_categorical or resampling_method == 'nearest':
        resample_alg = rasterio.enums.Resampling.nearest
    elif resampling_method == 'cubic':
        resample_alg = rasterio.enums.Resampling.cubic
    else:
        resample_alg = rasterio.enums.Resampling.bilinear

    data_type = 'uint8' if is_categorical else 'float32' # current canon dtype

    with rasterio.open(input_path) as src:

        nodata_val = (
            src.nodata
            if src.nodata is not None
            else (255 if is_categorical else -9999)
        )

        with rasterio.open(input_path) as src:
            with rasterio.vrt.WarpedVRT(
                src,
                crs=world_grid.crs,
                transform=world_grid.transform,
                width=world_grid.w,
                height=world_grid.h,
                resampling=resample_alg,
                nodata=nodata_val,
                dtype=data_type,
            ) as vrt:
                rasterio.shutil.copy(vrt, output_path, driver='VRT')

    return os.path.abspath(output_path)
