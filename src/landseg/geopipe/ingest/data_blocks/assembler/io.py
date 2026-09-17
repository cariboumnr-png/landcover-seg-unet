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
I/O helper utilities for block creation, padding, and integrity.

This module provides functions to interface with the filesystem and
read windowed NumPy arrays from source geospatial rasters using rasterio.
It handles multi-band extraction, reflection padding for DEM
neighborhoods, and compressed block load integrity verification.

Public APIs:
    - `RasterReadInput`: Dataclass specs parameter for reading rasters.
    - `RasterReadOutput`: Dataclass container for read raster arrays.
    - `read_band_map`: Return band-description to index mapping.
    - `read_label_specs`: Return per-band categorical label specs.
    - `read_schemes`: Return schemes dictionary embedded in a raster.
    - `check_npz_integrity`: Verify whether a .npz block file is valid.
    - `read_block_raster_data`: Read image/label and DEM bands.
'''

# standard imports
from __future__ import annotations
import ast
import dataclasses
import json
import typing
import zipfile
import zlib
# third-party imports
import numpy
import rasterio
import rasterio.errors
# local imports
import landseg.geopipe.core as geo_core
import landseg.geopipe.ingest.common.alias as alias
import landseg.geopipe.utils as geo_utils


# ----- public dataclasses
@dataclasses.dataclass(frozen=True)
class RasterReadInput:
    '''Specifications of parameters needed to read raster windows.'''
    image_fpath: str
    image_window: alias.RasterWindow
    image_band_map: dict[str, int]
    image_dem_pad_px: int
    label_fpath: str | None
    label_window: alias.RasterWindow | None
    label_specs: dict[str, geo_core.CategoricalSpec] | None


@dataclasses.dataclass(frozen=True)
class RasterReadOutput:
    '''Container holding loaded raster data arrays and metadata.'''
    image_array: numpy.ndarray
    image_padded_dem: numpy.ndarray | None
    image_nodata: float
    label_array: numpy.ndarray | None
    label_nodata: int | None


# ----- public functions
def read_band_map(fpath: str) -> dict[str, int]:
    '''
    Return lower-case band-description to zero-based index mapping.

    Args:
        fpath:
            File path to the raster dataset.

    Returns:
        dict[str, int]:
            Mapping from lowercase band description to zero-based index.
    '''
    try:
        with rasterio.open(fpath) as src:
            descriptions = src.descriptions
    except rasterio.errors.RasterioError:
        return {}

    if not descriptions:
        return {}

    names: list[str] = []

    for index, description in enumerate(descriptions):
        if description and description.strip():
            name = description.strip().lower()
        else:
            name = f'band_{index + 1}'

        names.append(name)

    if len(set(names)) != len(names):
        return {}

    return {name: index for index, name in enumerate(names)}


def read_label_specs(
    fpath: str | None,
) -> dict[str, geo_core.CategoricalSpec]:
    '''
    Return per-band label specifications embedded in a raster.

    Args:
        fpath:
            File path to raster dataset or None.

    Returns:
        dict[str, geo_core.CategoricalSpec]:
            Mapping from band name to categorical specifications.
    '''
    if fpath is None:
        return {}

    try:
        with rasterio.open(fpath) as src:
            descriptions = src.descriptions
            common_tags = src.tags() if src.count == 1 else {}

            specs = {}

            for index, description in enumerate(descriptions, start=1):
                tags = {**common_tags, **src.tags(index)}
                # required keys
                spec = {
                    key: _parse_vrt_tag(tags[key])
                    for key in ('num_cls', 'ignore_cls', 'index_base')
                }
                # optional keys
                for key in ('class_name', 'color_map', 'taxonomy'):
                    if key in tags:
                        value = _parse_vrt_tag(tags[key])
                        if value:
                            spec[key] = value

                name = (
                    description.strip()
                    if description and description.strip()
                    else f'band_{index}'
                )
                specs[name] = spec

            return specs

    except rasterio.errors.RasterioError as e:
        raise ValueError(f'Unable to read label specs from {fpath}') from e


def read_schemes(fpath: str | None) -> dict[str, typing.Any]:
    '''
    Return schemes dictionary embedded in a raster.

    Args:
        fpath:
            File path to raster dataset or None.

    Returns:
        dict[str, typing.Any]:
            Extracted schemes dictionary.
    '''
    if fpath is None:
        return {}

    def _merge_dict(target: dict, source: dict) -> None:
        for k, v in source.items():
            if isinstance(v, dict) and isinstance(target.get(k), dict):
                target[k].update(v)
            else:
                target[k] = v

    try:
        with rasterio.open(fpath) as src:
            all_schemes: dict[str, typing.Any] = {}
            # check dataset-level tags
            dataset_tags = src.tags()
            if 'schemes' in dataset_tags:
                val = _parse_vrt_tag(dataset_tags['schemes'])
                if isinstance(val, dict):
                    _merge_dict(all_schemes, val)

            # check per-band tags (for composite stacked VRTs)
            for b in src.indexes:
                band_tags = src.tags(b)
                if 'schemes' in band_tags:
                    val = _parse_vrt_tag(band_tags['schemes'])
                    if isinstance(val, dict):
                        _merge_dict(all_schemes, val)

            return all_schemes
    except (rasterio.errors.RasterioError, ValueError, SyntaxError):
        return {}


def check_npz_integrity(
    coord: tuple[int, int],
    fpath: str,
) -> dict[tuple[int, int], bool]:
    '''
    Verify whether a .npz block file can be successfully loaded.

    Args:
        coord:
            Grid coordinates being validated as (x, y).
        fpath:
            Path to the target .npz file.

    Returns:
        dict[tuple[int, int], bool]:
            Mapping from coordinate to boolean validity flag.
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
        inputs:
            RasterReadInput specifications defining window and paths.

    Returns:
        RasterReadOutput:
            Container holding read arrays, padded DEM, and nodata values.
    '''
    with geo_utils.open_rasters(
        inputs.image_fpath, inputs.label_fpath
    ) as (img, lbl):

        # a valid image array is required
        assert img, f'Invalid image source: {inputs.image_fpath}'
        if len(set(img.dtypes)) > 1:
            bands = [
                img.read(b, window=inputs.image_window, boundless=True).astype(
                    numpy.float32, copy=False
                )
                for b in range(1, img.count + 1)
            ]
            img_arr = numpy.stack(bands, axis=0)
        else:
            img_arr = img.read(window=inputs.image_window, boundless=True)
        image_nodata = img.nodata

        # read padded DEM if 'dem' is in band map
        padded_dem = None
        if 'dem' in inputs.image_band_map:
            dem_band = inputs.image_band_map['dem'] # 0-based
            padded_dem = _read_w_pad(
                img,
                inputs.image_window,
                dem_band + 1, # convert to 1-based for rasterio read
                inputs.image_dem_pad_px
            )

        # load label array if provided
        lbl_arr = None
        label_nodata = None
        if lbl is not None and inputs.label_window is not None:
            lbl_arr = lbl.read(window=inputs.label_window, boundless=True)
            assert isinstance(lbl_arr, numpy.ndarray)
            label_nodata = lbl.nodata

            if inputs.label_specs is not None:
                expected_bands = len(inputs.label_specs)
                if lbl_arr.shape[0] != expected_bands:
                    raise ValueError (
                        f'Label targets number != input label array shape '
                        f'on axis 0: {lbl_arr.shape[0]} != {expected_bands}'
                    )

        return RasterReadOutput(
            image_array=img_arr,
            image_padded_dem=padded_dem,
            image_nodata=image_nodata,
            label_array=lbl_arr,
            label_nodata=label_nodata
        )


# ----- private helpers
def _parse_vrt_tag(value: str) -> object:
    '''Decode GDAL metadata serialized as JSON or a Python literal.'''
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return ast.literal_eval(value)


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
        ) # type: ignore
    except ValueError as e:
        raise ValueError(
            f'Error reading DEM with pad ({pad}), padded raster window: '
            f'{nw_x}, {nw_y}, {se_x - nw_x}, {se_y - nw_y}'
        ) from e

    # get required padding on each side - no padding if within raster bound
    pad_top = max(0, pad - window.row_off)
    pad_left = max(0, pad - window.col_off)
    pad_bottom = max(0, (window.row_off + window.height + pad) - img.height)
    pad_right = max(0, (window.col_off + window.width + pad) - img.width)

    # pad the expanded array accordingly controlled by pad_width and return
    expanded_padded = numpy.pad(
        img.read(dem_band, window=_window),                 # array
        ((pad_top, pad_bottom), (pad_left, pad_right)),     # pad_width
        'reflect'                                           # mode
    )
    return expanded_padded
