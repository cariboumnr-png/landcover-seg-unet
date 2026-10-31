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
Spatial grid specification and raster warping operations.

This module provides functions to reproject, resample, and snap input
rasters to a canonical world grid layout as GDAL Virtual Rasters (VRT).

Public APIs:
    - `is_same_crs`: Compare two coordinate reference systems for equality.
    - `resolve_resampling_algorithm`: Resolve rasterio resampling enum.
    - `write_warped_vrt`: Write a virtual warped raster (VRT) to disk.
'''

# standard imports
import pathlib
# third-party imports
import rasterio
import rasterio.crs
import rasterio.enums
import rasterio.errors
import rasterio.shutil
import rasterio.vrt
# local imports
import landseg.geopipe.rasters._types as t
import landseg.geopipe.rasters.io as io


# ----- public functions
def is_same_crs(
    crs1: t.CRS | str | None,
    crs2: t.CRS | str | None,
) -> bool:
    '''Compare two coordinate reference systems for equality.'''
    if crs1 is None or crs2 is None:
        return crs1 == crs2
    try:
        c1 = rasterio.crs.CRS.from_user_input(crs1)
        c2 = rasterio.crs.CRS.from_user_input(crs2)
        return c1 == c2
    except (rasterio.errors.CRSError, ValueError):
        return str(crs1).strip() == str(crs2).strip()


def write_warped_vrt( # pylint: disable=too-many-arguments
    *,
    src: pathlib.Path | str | t.DatasetReader,
    output_path: pathlib.Path | str,
    crs: t.CRS,
    transform: t.Affine,
    width: int,
    height: int,
    resampling: t.Resampling,
    nodata: float | int | None = None,
    dtype: str | None = None,
) -> pathlib.Path:
    '''
    Write a virtual warped raster (VRT) to disk.

    Args:
        src:
            Open source dataset reader or path to raster file.
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
        dtype:
            Optional dtype value.

    Returns:
        pathlib.Path:
            Resolved absolute path to the created warped VRT file.
    '''
    out_p = pathlib.Path(output_path).resolve()
    out_p.parent.mkdir(parents=True, exist_ok=True)

    if isinstance(src, (str, pathlib.Path)):
        with io.read_raster(src) as reader:
            return _write_warped_vrt(
                reader=reader,
                out_p=out_p,
                crs=crs,
                transform=transform,
                width=width,
                height=height,
                resampling=resampling,
                nodata=nodata,
                dtype=dtype,
            )

    return _write_warped_vrt(
        reader=src,
        out_p=out_p,
        crs=crs,
        transform=transform,
        width=width,
        height=height,
        resampling=resampling,
        nodata=nodata,
        dtype=dtype,
    )


def resolve_resampling_algorithm(
    method_name: str,
    is_categorical: bool,
) -> t.Resampling:
    '''Resolve rasterio resampling algorithm enum.'''
    if is_categorical or method_name == 'nearest':
        return rasterio.enums.Resampling.nearest
    if method_name == 'cubic':
        return rasterio.enums.Resampling.cubic
    try:
        return rasterio.enums.Resampling[method_name.lower()]
    except KeyError:
        return rasterio.enums.Resampling.bilinear


# ----- private helpers
def _write_warped_vrt( # pylint: disable=too-many-arguments
    *,
    reader: t.DatasetReader,
    out_p: pathlib.Path,
    crs: t.CRS,
    transform: t.Affine,
    width: int,
    height: int,
    resampling: t.Resampling,
    nodata: float | int | None,
    dtype: str | None,
) -> pathlib.Path:
    '''Warp an open reader to disk as a VRT file.'''
    effective_nodata = reader.nodata if nodata is None else nodata
    with rasterio.vrt.WarpedVRT(
        reader,
        crs=crs,
        transform=transform,
        width=width,
        height=height,
        resampling=resampling,
        nodata=effective_nodata,
        dtype=dtype,
    ) as warped:
        rasterio.shutil.copy(warped, out_p, driver='VRT')
    return out_p
