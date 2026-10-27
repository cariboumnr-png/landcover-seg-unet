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
import dataclasses
import pathlib
import typing
import zipfile
import zlib
# local imports
import landseg.geopipe.core as geo_core
import landseg.geopipe.rasters as rasters

if typing.TYPE_CHECKING:
    import numpy.typing


# ----- public dataclasses
@dataclasses.dataclass(frozen=True)
class RasterReadInput:
    '''Specifications of parameters needed to read raster windows.'''
    image_fpath: pathlib.Path
    image_window: rasters.Window
    image_band_map: dict[str, int]
    image_dem_pad_px: int
    label_fpath: pathlib.Path | None
    label_window: rasters.Window | None
    label_specs: dict[str, geo_core.CategoricalSpec] | None


@dataclasses.dataclass(frozen=True)
class RasterReadOutput:
    '''Container holding loaded raster data arrays and metadata.'''
    image_array: numpy.typing.NDArray
    image_padded_dem: numpy.typing.NDArray | None
    image_nodata: float
    label_array: numpy.typing.NDArray | None
    label_nodata: int | None


# ----- public functions
def read_band_map(fpath: pathlib.Path) -> dict[str, int]:
    '''
    Return lower-case band-description to zero-based index mapping.

    Args:
        fpath:
            File path to the raster dataset.

    Returns:
        dict[str, int]:
            Mapping from lowercase band description to zero-based index.
    '''
    raster_info = rasters.get_info(fpath)
    return raster_info.band_map


def read_label_specs(
    fpath: pathlib.Path | None,
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

    raster_info = rasters.get_info(fpath)

    descriptions = raster_info.descriptions
    dataset_tags = raster_info.tags if raster_info.count == 1 else {}
    # include dataset-level tags if raster is 1-band

    specs = {}

    for i, desc in enumerate(descriptions):
        tags = {**dataset_tags, **raster_info.band_tags[i]}
        # required keys
        spec = {
            key: rasters.parse_vrt_tag(tags[key])
            for key in ('num_cls', 'ignore_cls', 'index_base')
        }
        if isinstance(spec.get('ignore_cls'), (list, tuple)):
            spec['ignore_cls'] = tuple(spec['ignore_cls'])

        # optional keys
        for key in ('class_name', 'color_map', 'taxonomy'):
            if key in tags:
                value = rasters.parse_vrt_tag(tags[key])
                if value:
                    if key == 'color_map' and isinstance(value, dict):
                        value = {
                            k: tuple(rgb) if isinstance(rgb, (list, tuple))
                            else rgb
                            for k, rgb in value.items()
                        }
                    spec[key] = value

        name = desc.strip() if desc and desc.strip() else f'band_{i}'
        specs[name] = spec

    return specs


def read_schemes(fpath: pathlib.Path | None) -> dict[str, typing.Any]:
    '''
    Return schemes dictionary embedded in a raster.

    Args:
        fpath:
            File path to raster dataset or None.

    Returns:
        dict[str, typing.Any]:
            Extracted schemes dictionary.
    '''

    def _merge_dict(target: dict, source: dict) -> None:
        for k, v in source.items():
            if isinstance(v, dict) and isinstance(target.get(k), dict):
                target[k].update(v)
            else:
                target[k] = v

    if fpath is None:
        return {}

    raster_info = rasters.get_info(fpath)
    all_schemes: dict[str, typing.Any] = {}

    # check dataset-level tags
    dataset_tags = raster_info.tags
    if 'schemes' in dataset_tags:
        val = rasters.parse_vrt_tag(dataset_tags['schemes'])
        if isinstance(val, dict):
            _merge_dict(all_schemes, val)

    # check per-band tags (for composite stacked VRTs)
    for band_tags in raster_info.band_tags:
        if 'schemes' in band_tags:
            val = rasters.parse_vrt_tag(band_tags['schemes'])
            if isinstance(val, dict):
                _merge_dict(all_schemes, val)

    return all_schemes

def check_npz_integrity(
    coord: tuple[int, int],
    fpath: pathlib.Path,
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
    padded_dem = None
    lbl_arr = None
    lbl_nodata = None

    with rasters.read_raster(inputs.image_fpath) as image_reader:
        img_nodata = image_reader.nodata

        img_arr = rasters.read_window(
            image_reader,
            inputs.image_window,
            edge_mode='boundless',
        ).astype('float32', copy=False)

        if 'dem' in inputs.image_band_map:
            dem_band = inputs.image_band_map['dem'] # 0-based
            padded_dem = rasters.read_window(
                image_reader,
                inputs.image_window,
                indexes=dem_band + 1, # convert to 1-based for rasterio read
                edge_mode='reflect',
                padding=inputs.image_dem_pad_px,
            ).astype('float32', copy=False)


    if inputs.label_fpath is not None and inputs.label_window is not None:
        with rasters.read_raster(inputs.label_fpath) as label_reader:
            lbl_nodata = label_reader.nodata

            lbl_arr = rasters.read_window(
                label_reader,
                inputs.label_window,
            ).astype('uint8', copy=False)

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
        image_nodata=img_nodata,
        label_array=lbl_arr,
        label_nodata=lbl_nodata
    )
