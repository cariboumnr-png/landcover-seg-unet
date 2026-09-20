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
Multi-raster channel composition and nodata mask unification operations.

This module provides utilities to generate a unified valid-pixel boolean
mask across all bands of a raster.

Public APIs:
    - `unify_nodata_mask`: Create a 1-band valid pixel mask across bands.
'''

# standard imports
from __future__ import annotations
import os
# third-party imports
import numpy
import rasterio
import rasterio.shutil
import rasterio.vrt


# ----- public functions
def unify_nodata_mask(
    input_path: str,
    output_mask_path: str,
) -> str:
    '''
    Create a 1-band boolean valid pixel mask across bands.

    Value mapping: 1=valid, 0=nodata.

    Args:
        input_path:
            Path to the multi-band source raster.
        output_mask_path:
            Destination path for the valid-pixel mask raster.

    Returns:
        str:
            Absolute path to the created mask raster.
    '''
    out_dir = os.path.dirname(os.path.abspath(output_mask_path))
    os.makedirs(out_dir, exist_ok=True)

    with rasterio.open(input_path) as src:
        valid_mask = numpy.ones((src.height, src.width), dtype=numpy.uint8)

        for b in range(1, src.count + 1):
            data = src.read(b)
            if src.nodatavals and src.nodatavals[b - 1] is not None:
                nodata_val = src.nodatavals[b - 1]
                if numpy.isnan(nodata_val):
                    valid_mask[numpy.isnan(data)] = 0
                else:
                    valid_mask[data == nodata_val] = 0

        meta = src.meta.copy()
        meta.update({
            'count': 1,
            'dtype': 'uint8',
            'nodata': 0,
            'compress': 'deflate'
        })
        if output_mask_path.endswith('.vrt'):
            meta['driver'] = 'GTiff'
            raw_tif = output_mask_path[:-4] + '_mask.tif'
            with rasterio.open(raw_tif, 'w', **meta) as dst:
                dst.write(valid_mask, 1)

            with rasterio.open(raw_tif) as mask_src:
                with rasterio.vrt.WarpedVRT(mask_src) as vrt:
                    rasterio.shutil.copy(vrt, output_mask_path, driver='VRT')
        else:
            meta['driver'] = 'GTiff'
            with rasterio.open(output_mask_path, 'w', **meta) as dst:
                dst.write(valid_mask, 1)

    return os.path.abspath(output_mask_path)
