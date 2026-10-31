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
I/O helper utilities for block creation, padding, and integrity.

This module provides functions to interface with the filesystem and
read windowed NumPy arrays from source geospatial rasters using rasterio.
It handles multi-band extraction, reflection padding for DEM
neighborhoods, and compressed block load integrity verification.

Public APIs:
    - `RasterReadContext`: Dataclass specs parameter for reading rasters.
    - `RasterReadOutput`: Dataclass container for read raster arrays.
    - `read_block_raster_data`: Read image/label and DEM bands.
    - `read_schemes`: Return schemes dictionary embedded in a raster.
    - `check_npz_integrity`: Verify whether a .npz block file is valid.
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
class RasterReadContext:
    '''Immutable context for raster reading by window.'''
    image_fpath: pathlib.Path
    label_fpath: pathlib.Path | None
    image_band_map: dict[str, int]
    label_specs: dict[str, geo_core.CategoricalSpec] | None

    @property
    def label_color_map(self) -> dict[str, tuple[int, int, int]] | None:
        '''Fetch and retrieve label color map if provided.'''
        label_color_map: dict[str, tuple[int, int, int]] | None = None
        if self.label_specs:
            for spec in self.label_specs.values():
                if 'color_map' in spec and spec['color_map']:
                    label_color_map = spec['color_map']
                    break
        return label_color_map

    @classmethod
    def from_rasters(
        cls,
        image_fpath: pathlib.Path,
        label_fpath: pathlib.Path | None = None,
        *,
        image_band_map: dict[str, int] | None = None,
        label_specs: dict[str, geo_core.CategoricalSpec] | None = None,
    ) -> typing.Self:
        '''Construct the context from input rasters.'''
        return cls(
            image_fpath=image_fpath,
            label_fpath=label_fpath,
            image_band_map=(
                image_band_map
                if image_band_map is not None
                else _get_img_band_map(image_fpath)
            ),
            label_specs=(
                label_specs
                if label_specs is not None
                else (_get_lbl_specs(label_fpath) if label_fpath else None)
            ),
        )

    def validate_feature_specs(
        self,
        add_spectral: list[str] | None = None,
        add_topo: list[str] | None = None,
    ):
        '''Validate if feature engineering choices are feasible.'''
        band_map = [b.lower() for b in self.image_band_map]

        if add_spectral:
            spectral = [s.lower() for s in add_spectral]
            invalid = [s for s in spectral if s not in ['ndvi', 'ndmi', 'nbr']]
            if invalid:
                raise ValueError(f'Invalid spectral indices: {invalid}')

            if 'red' not in band_map:
                raise ValueError('Unable to add spectrals: red band missing')
            if 'ndvi' in spectral and 'nir' not in band_map:
                raise ValueError('NDVI calculation: NIR band missing')
            if 'ndmi' in spectral and 'swir1' not in band_map:
                raise ValueError('NDMI calculation: SWIR1 band missing')
            if 'nbr' in spectral and 'swir2' not in band_map:
                raise ValueError('NBR calculation: SWIR2 band missing')

        if add_topo:
            topo = [t.lower() for t in add_topo]
            invalid = [t for t in topo if t not in ['slope', 'aspect', 'tpi']]
            if invalid:
                raise ValueError(f'Invalid topo features: {invalid}')

            if 'dem' not in band_map:
                raise ValueError('DEM band missing for topographical features')

        if self.label_fpath and not self.label_specs:
            raise ValueError('"label_specs" not provided')


@dataclasses.dataclass(frozen=True)
class RasterReadOutput:
    '''Container holding loaded raster data arrays and metadata.'''
    image_array: numpy.typing.NDArray
    image_nodata: float
    image_band_map: dict[str, int]
    image_padded_dem: numpy.typing.NDArray | None
    label_array: numpy.typing.NDArray | None
    label_nodata: int | None
    label_specs: dict[str, geo_core.CategoricalSpec] | None

    @property
    def pad_dem(self) -> numpy.ndarray:
        '''Return padded DEM array if provided.'''
        if self.image_padded_dem is None:
            raise ValueError('Cannot access padded DEM as it is not provided')
        return self.image_padded_dem


# ----- public functions
def read_block_raster_data(
    image_window: rasters.Window,
    label_window: rasters.Window | None,
    context: RasterReadContext,
    *,
    dem_pad: int = 8,
) -> RasterReadOutput:
    '''
    Read arrays and metadata from raster datasets for a given window.

    Args:
        image_window:
            Window bounding box for the image raster.
        label_window:
            Window bounding box for the label raster, if present.
        context:
            Raster read context containing paths and metadata.
        dem_pad:
            Padding in pixels for DEM neighborhood reflection.

    Returns:
        RasterReadOutput:
            Container holding read arrays, padded DEM, and nodata values.
    '''
    # read image raster
    padded_dem = None
    with rasters.read_raster(context.image_fpath) as image_reader:
        img_nodata = image_reader.nodata

        img_arr = rasters.read_window(
            image_reader,
            image_window,
            edge_mode='boundless',
        ).astype('float32', copy=False)

        if img_arr.ndim != 3:
            raise ValueError('Image array is not of shape [C, H, W]')

        if 'dem' in context.image_band_map: # 0-based
            padded_dem = rasters.read_window(
                image_reader,
                image_window,
                indexes=context.image_band_map['dem'] + 1, # to 1-based
                edge_mode='reflect',
                padding=dem_pad,
            ).astype('float32', copy=False)

    # read label raster if provided
    lbl_arr = None
    lbl_nodata = None
    if label_window is not None:
        assert context.label_fpath is not None, 'label raster not provided'
        assert context.label_specs is not None, 'label specs not provided'
        with rasters.read_raster(context.label_fpath) as label_reader:
            lbl_nodata = label_reader.nodata

            lbl_arr = rasters.read_window(
                label_reader,
                label_window,
            ).astype('uint8', copy=False)

            if lbl_arr.ndim != 3:
                raise ValueError('Label array is not of shape [C, H, W]')
            if img_arr.shape[-2:] != lbl_arr.shape[-2:]:
                raise ValueError('Image and label arrays have different H / W')

            expected_bands = len(context.label_specs)
            if lbl_arr.shape[0] != expected_bands:
                raise ValueError (
                    f'Label targets number != input label array shape '
                    f'on axis 0: {lbl_arr.shape[0]} != {expected_bands}'
                )

    return RasterReadOutput(
        image_array=img_arr,
        image_band_map=context.image_band_map,
        image_padded_dem=padded_dem,
        image_nodata=img_nodata,
        label_array=lbl_arr,
        label_nodata=lbl_nodata,
        label_specs=context.label_specs,
    )


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


# ----- private helpers
def _get_img_band_map(fpath: pathlib.Path) -> dict[str, int]:
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


def _get_lbl_specs(fpath: pathlib.Path) -> dict[str, geo_core.CategoricalSpec]:
    '''
    Return per-band label specifications embedded in a raster.

    Args:
        fpath:
            File path to raster dataset or None.

    Returns:
        dict[str, geo_core.CategoricalSpec]:
            Mapping from band name to categorical specifications.
    '''
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
