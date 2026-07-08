# =========================================================================== #
#           Copyright (c) His Majesty the King in right of Ontario,           #
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
I/O helper utilities for block creation, padding, and integrity.

This module provides focused functions to interface with the file
system and read windowed NumPy arrays from source geospatial rasters
(e.g. TIFFs) using `rasterio`. It abstracts away multi-band
extraction, coordinate-offset math, reflection padding for DEM
neighborhoods, and compressed block load integrity verification.

Public APIs:
    - RasterReadInput: Dataclass specs parameter for reading rasters.
    - RasterReadOutput: Dataclass container for read raster arrays.
    - check_npz_integrity: Verifies a saved .npz file is readable.
    - read_block_raster_data: Reads image/label and DEM bands.
'''

# standard imports
import dataclasses
import zipfile
import zlib
# third-party imports
import numpy
# local imports
import landseg.geopipe.core as geo_core
import landseg.geopipe.foundation.common.alias as alias
import landseg.geopipe.utils as geo_utils


@dataclasses.dataclass(frozen=True)
class RasterReadInput:
    '''Specifications of parameters needed to read raster windows.'''
    image_fpath: str
    label_fpath: str | None
    image_window: alias.RasterWindow
    label_window: alias.RasterWindow | None
    dem_pad_px: int
    image_band_map: dict[str, int]
    label_specs: dict[str, geo_core.LabelSpecs] | None


@dataclasses.dataclass(frozen=True)
class RasterReadOutput:
    '''Containers holding loaded raster data arrays and metadata.'''
    image_array: numpy.ndarray
    label_array: numpy.ndarray | None
    padded_dem: numpy.ndarray | None
    image_nodata: float
    label_nodata: float | None


def check_npz_integrity(
    coord: tuple[int, int],
    fpath: str,
) -> dict[tuple[int, int], bool]:
    '''
    Verify whether a `.npz` block file can be successfully loaded.

    Args:
        coord: The grid coordinates being validated.
        fpath: Path to the target `.npz` file.

    Returns:
        dict: A mapping {coord: is_valid} where is_valid is True
            if the file was loaded successfully; False otherwise.
    '''
    ok = False
    try:
        geo_core.DataBlock.load(fpath)
        ok = True
    except (FileNotFoundError, zipfile.error, zlib.error, ValueError):
        ok = False
    return {coord: ok}


def read_block_raster_data(inputs: RasterReadInput) -> RasterReadOutput:
    '''
    Read arrays and metadata from raster datasets for a given window.

    Args:
        inputs: RasterReadInput specs.

    Returns:
        RasterReadOutput: Containing read arrays and nodata values.
    '''
    with geo_utils.open_rasters(
        inputs.image_fpath, inputs.label_fpath
    ) as (img, lbl):
        assert img, f'Invalid image source: {inputs.image_fpath}'

        img_arr = img.read(window=inputs.image_window, boundless=True)
        image_nodata = img.nodata

        # read padded DEM if 'dem' is in band map
        padded_dem = None
        if 'dem' in inputs.image_band_map:
            dem_band = inputs.image_band_map['dem']
            padded_dem = _read_w_pad(
                img,
                inputs.image_window,
                dem_band,
                inputs.dem_pad_px
            )

        lbl_arr = None
        label_nodata = None
        if lbl is not None and inputs.label_window is not None:
            lbl_arr = lbl.read(window=inputs.label_window, boundless=True)
            assert isinstance(lbl_arr, numpy.ndarray)
            label_nodata = lbl.nodata

            if inputs.label_specs is not None:
                expected_bands = len(inputs.label_specs)
                if lbl_arr.shape[0] != expected_bands:
                    msg = (
                        f'Label targets number != input label array shape '
                        f'on axis 0: {lbl_arr.shape[0]} != {expected_bands}'
                    )
                    raise ValueError(msg)

        return RasterReadOutput(
            image_array=img_arr,
            label_array=lbl_arr,
            padded_dem=padded_dem,
            image_nodata=image_nodata,
            label_nodata=label_nodata
        )


def _read_w_pad(
    img: alias.RasterReader,
    window: alias.RasterWindow,
    dem_band: int,
    pad: int
) -> numpy.ndarray:
    '''Read the DEM band around 'window' with reflection padding.'''
    # expand window within the original raster
    nw_x = max(window.col_off - pad, 0)
    nw_y = max(window.row_off - pad, 0)
    se_x = min(window.col_off + window.width + pad, img.width)
    se_y = min(window.row_off + window.height + pad, img.height)
    try:
        _window = alias.RasterWindow(
            nw_x, nw_y, se_x - nw_x, se_y - nw_y
        )  # type: ignore
    except ValueError:
        print(nw_x, nw_y, se_x - nw_x, se_y - nw_y)
        raise

    # get expanded array using the new window
    # band number in rasterio.read is 1-based
    expanded = img.read(dem_band + 1, window=_window)

    # get required padding on each side - no padding if within raster bound
    pad_top = max(0, pad - window.row_off)
    pad_left = max(0, pad - window.col_off)
    pad_bottom = max(0, (window.row_off + window.height + pad) - img.height)
    pad_right = max(0, (window.col_off + window.width + pad) - img.width)

    # pad the expanded arr accordingly controlled by pad_width and return
    expanded_padded = numpy.pad(
        array=expanded,
        pad_width=((pad_top, pad_bottom), (pad_left, pad_right)),
        mode='reflect'
    )
    return expanded_padded
