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
Multi-raster channel composition and nodata mask unification operations.

This module provides utilities to generate a unified valid-pixel boolean
mask across all bands of a raster.

Public APIs:
    - `unify_nodata_mask`: Create a 1-band valid pixel mask across bands.
'''

# standard imports
import pathlib
# third-party imports
import numpy
import rasterio
import rasterio.shutil
import rasterio.vrt
# local imports
import landseg.artifacts.storage.scratch as scratch


# ----- public functions
def unify_nodata_mask(
    input_path: pathlib.Path,
    output_mask_path: pathlib.Path,
    *,
    scratch_dir: str | pathlib.Path | None = None,
) -> pathlib.Path:
    '''
    Create a 1-band boolean valid pixel mask across bands.

    Streams block-by-block using source block windows to bound RAM
    usage to tile dimensions, writing to a tiled GeoTIFF staged via
    scratch storage.

    Value mapping: 1=valid, 0=nodata.

    Args:
        input_path:
            Path to the multi-band source raster.
        output_mask_path:
            Destination path for the valid-pixel mask raster (.tif or .vrt).
        scratch_dir:
            Optional scratch directory override.

    Returns:
        pathlib.Path:
            Absolute path to the created mask raster.
    '''
    out_path = pathlib.Path(output_mask_path).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    is_vrt = out_path.suffix == '.vrt'
    target_tif = (
        out_path.with_name(f'{out_path.stem}_mask.tif')
        if is_vrt
        else out_path
    )

    with rasterio.open(input_path) as src:
        meta = src.meta.copy()
        meta.update({
            'count': 1,
            'dtype': 'uint8',
            'nodata': 0,
            'driver': 'GTiff',
            'tiled': True,
            'blockxsize': 256,
            'blockysize': 256,
            'compress': 'deflate',
        })

        nodatavals = src.nodatavals or [None] * src.count
        same_dtype = len(set(src.dtypes)) <= 1

        with scratch.stage_to_scratch(
            target_tif,
            scratch_dir=scratch_dir,
        ) as stage_f:
            with rasterio.open(stage_f, 'w', **meta) as dst:
                for _, window in src.block_windows(1):
                    win_h = int(window.height)
                    win_w = int(window.width)
                    mask_block = numpy.ones((win_h, win_w), dtype=numpy.uint8)

                    if same_dtype:
                        block_data = src.read(window=window)
                        for b_idx in range(src.count):
                            band_data = block_data[b_idx]
                            nodata_val = nodatavals[b_idx]
                            if nodata_val is not None:
                                if numpy.isnan(nodata_val):
                                    mask_block[numpy.isnan(band_data)] = 0
                                else:
                                    mask_block[band_data == nodata_val] = 0
                    else:
                        for b in range(1, src.count + 1):
                            band_data = src.read(b, window=window)
                            nodata_val = nodatavals[b - 1]
                            if nodata_val is not None:
                                if numpy.isnan(nodata_val):
                                    mask_block[numpy.isnan(band_data)] = 0
                                else:
                                    mask_block[band_data == nodata_val] = 0

                    dst.write(mask_block, 1, window=window)

        if is_vrt:
            with rasterio.open(target_tif) as mask_src:
                with rasterio.vrt.WarpedVRT(mask_src) as vrt:
                    rasterio.shutil.copy(vrt, out_path, driver='VRT')

    return out_path
