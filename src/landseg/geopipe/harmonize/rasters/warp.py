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
import pathlib
# third-party imports
import rasterio
import rasterio.crs
import rasterio.enums
import rasterio.shutil
import rasterio.transform
import rasterio.vrt
# local imports
import landseg.geopipe.core as geo_core


# ----- public functions
def warp_to_grid(
    *,
    input_path: pathlib.Path,
    output_path: pathlib.Path,
    world_grid: geo_core.GridLayout,
    resampling_method: str,
    is_categorical: bool = False,
) -> pathlib.Path:
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
        pathlib.Path:
            Absolute path to output harmonized Virtual Raster file.
    '''
    out_path = pathlib.Path(output_path).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    resample_alg = resolve_resampling_algorithm(
        resampling_method,
        is_categorical
    )

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

    return out_path


def write_warped_vrt( # pylint: disable=too-many-arguments
    *,
    src: rasterio.DatasetReader,
    output_path: pathlib.Path | str,
    crs: rasterio.crs.CRS,
    transform: rasterio.transform.Affine,
    width: int,
    height: int,
    resampling: rasterio.enums.Resampling = rasterio.enums.Resampling.bilinear,
    nodata: float | int | None = None,
) -> pathlib.Path:
    '''
    Write a virtual warped raster (VRT) to disk.

    Args:
        src:
            Open source dataset reader.
        output_path:
            Target destination path for the warped .vrt file.
        crs:
            Target coordinate reference system.
        transform:
            Target affine transform.
        width:
            Target width in pixels.
        height:
            Target height in pixels.
        resampling:
            Resampling algorithm enum.
        nodata:
            Optional nodata value.

    Returns:
        pathlib.Path:
            Resolved absolute path to the created warped VRT file.
    '''
    out_p = pathlib.Path(output_path).resolve()
    out_p.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.vrt.WarpedVRT(
        src,
        crs=crs,
        transform=transform,
        width=width,
        height=height,
        resampling=resampling,
        nodata=nodata,
    ) as warped:
        rasterio.shutil.copy(warped, out_p, driver='VRT')
    return out_p


def resolve_resampling_algorithm(
    method_name: str,
    is_categorical: bool,
) -> rasterio.enums.Resampling:
    '''Resolve rasterio resampling algorithm enum.'''
    if is_categorical or method_name == 'nearest':
        return rasterio.enums.Resampling.nearest
    if method_name == 'cubic':
        return rasterio.enums.Resampling.cubic
    try:
        return rasterio.enums.Resampling[method_name.lower()]
    except KeyError:
        return rasterio.enums.Resampling.bilinear
