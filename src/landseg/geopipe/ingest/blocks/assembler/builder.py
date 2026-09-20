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
Feature engineering and DataBlock assembly pipeline for geospatial rasters.

Processes raw windowed imagery and label arrays, generating derived
spectral indices, topographic metrics from elevation models, canonical
multi-head label stacks, and initial per-block statistics into a
materialized DataBlock.

Public APIs:
    - DataBlockInputs: container for raw arrays to construct a DataBlock.
    - DataBlockConfig: build-time configuration for feature engineering.
    - build_data_block: construct a DataBlock with derived features.
'''

# standard imports
from __future__ import annotations
import dataclasses
import math
# third-party imports
import numpy
# local imports
import landseg.geopipe.core as geo_core


# ----- public dataclasses
@dataclasses.dataclass
class DataBlockInputs:
    '''Container for source materials needed to build a `DataBlock`.'''
    block_name: str
    image_array: numpy.ndarray
    image_padded_dem: numpy.ndarray | None
    label_array: numpy.ndarray | None

    def __post_init__(self):
        if self.image_array.ndim != 3:
            raise ValueError('Image array is not of shape [C, H, W]')

        if self.label_array is not None:
            if self.label_array.ndim != 3:
                raise ValueError('Label array is not of shape [C, H, W]')
            if self.image_array.shape[-2:] != self.label_array.shape[-2:]:
                raise ValueError('Image and label arrays have different H / W')

    @property
    def has_label(self) -> bool:
        '''Return `True` if label array is provided.'''
        return self.label_array is not None

    @property
    def pad_dem(self) -> numpy.ndarray:
        '''Return padded DEM array if provided.'''
        if self.image_padded_dem is None:
            raise ValueError('Cannot access padded DEM as it is not provided')
        return self.image_padded_dem


@dataclasses.dataclass(frozen=True)
class DataBlockConfig:
    '''Build-time config for feature engineering and data encoding.'''
    image_nodata: float
    image_band_map: dict[str, int]
    image_dem_pad_px: int
    label_nodata: int | None = None
    label_specs: dict[str, geo_core.CategoricalSpec] | None = None
    label_ignore_index: int = 255
    add_spectral: list[str] | None = None
    add_topo: list[str] | None = None

    def __post_init__(self):
        band_map = [b.lower() for b in self.image_band_map]

        if self.add_spectral:
            spectral = [s.lower() for s in self.add_spectral]
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

        if self.add_topo:
            topo = [t.lower() for t in self.add_topo]
            invalid = [t for t in topo if t not in ['slope', 'aspect', 'tpi']]
            if invalid:
                raise ValueError(f'Invalid topo features: {invalid}')

            if 'dem' not in band_map:
                raise ValueError('DEM band missing for topographical features')


# ----- public functions
def build_data_block(
    inputs: DataBlockInputs,
    config: DataBlockConfig,
) -> geo_core.DataBlock:
    '''
    Construct a DataBlock from source arrays and build config.

    Executes the feature engineering pipeline, including spectral
    index computation, topographic metrics, label canonicalization,
    valid mask generation, and per-band statistical summaries.

    Args:
        inputs:
            source arrays and metadata required to build the block.
        config:
            build configuration controlling feature engineering.

    Returns:
        geo_core.DataBlock:
            a fully populated block instance.
    '''
    manifest = geo_core.DataBlock.empty_manifest(inputs.block_name)
    manifest['image_band_map'] = dict(config.image_band_map)
    manifest['image_nodata'] = config.image_nodata
    manifest['label_ignore_index'] = config.label_ignore_index
    manifest['label_nodata'] = config.label_nodata or 0

    image = inputs.image_array.astype(numpy.float32)

    if config.add_spectral:
        image = _image_add_spectral(
            image,
            manifest,
            [item.lower() for item in config.add_spectral],
        )

    if config.add_topo:
        padded_dem = inputs.pad_dem.astype(numpy.float32)
        image = _image_add_topography(
            image,
            padded_dem,
            manifest,
            [item.lower() for item in config.add_topo],
            config.image_dem_pad_px,
        )

    valid_mask, img_ratio = _image_get_valid_mask(
        image, manifest['image_nodata']
    )
    manifest['valid_ratios']['image'] = img_ratio
    manifest['image_stats'] = _image_get_stats(
        image, manifest['image_nodata']
    )

    if inputs.has_label:
        if not config.label_specs:
            raise ValueError('"label_specs" not provided')
        manifest['has_label'] = True
        manifest['label_nodata'] = config.label_nodata or -1
        assert inputs.label_array is not None
        raw_label = inputs.label_array.astype(numpy.uint8)
        label_stack = _label_canonicalize(
            raw_label,
            config.label_specs,
            manifest,
        )
    else:
        label_stack = numpy.array([1])
        manifest['has_label'] = False

    arrays = geo_core.DataBlockArrays(
        image=image,
        label=label_stack,
        valid_mask=valid_mask,
    )
    arrays.validate()
    return geo_core.DataBlock(data=arrays, manifest=manifest)


# ----- private helpers
def _image_add_spectral(
    image: numpy.ndarray,
    manifest: geo_core.DataBlockManifest,
    indices: list[str],
) -> numpy.ndarray:
    '''Add spectral indices if related bands are available.'''
    band_idx = manifest['image_band_map']
    nodata = manifest['image_nodata']
    red = _Calc.mask(image[band_idx['red']], nodata)
    spectrals: list[numpy.ndarray] = []
    next_idx = image.shape[0]

    if 'ndvi' in indices:
        nir = _Calc.mask(image[band_idx['nir']], nodata)
        spectrals.append(_Calc.ndvi(nir, red, nodata))
        band_idx['ndvi'] = next_idx
        next_idx += 1

    if 'ndmi' in indices:
        nir = _Calc.mask(image[band_idx['nir']], nodata)
        swir1 = _Calc.mask(image[band_idx['swir1']], nodata)
        spectrals.append(_Calc.ndmi(nir, swir1, nodata))
        band_idx['ndmi'] = next_idx
        next_idx += 1

    if 'nbr' in indices:
        nir = _Calc.mask(image[band_idx['nir']], nodata)
        swir2 = _Calc.mask(image[band_idx['swir2']], nodata)
        spectrals.append(_Calc.nbr(nir, swir2, nodata))
        band_idx['nbr'] = next_idx
        next_idx += 1

    if spectrals:
        added = numpy.stack(spectrals).astype(numpy.float32)
        return numpy.append(image, added, axis=0)
    return image


def _compute_topo_windows(
    padded_dem: numpy.ndarray,
    arrs: dict[str, numpy.ndarray],
    features: list[str],
    pad: int,
    nodata: float,
) -> None:
    '''Populate topography metric arrays for each pixel window.'''
    max_h, max_w = padded_dem.shape
    for y in range(pad, max_h - pad):
        for x in range(pad, max_w - pad):
            if 'slope' in features:
                pxs = _Calc.get_px_group(padded_dem, x, y, 1)
                assert pxs.shape == (3, 3)
                (
                    arrs['slope'][y - pad, x - pad],
                    arrs['cos_a'][y - pad, x - pad],
                    arrs['sin_a'][y - pad, x - pad]
                ) = _Calc.slope_n_aspect(pxs, nodata)

            if 'tpi' in features:
                pxs = _Calc.get_px_group(padded_dem, x, y, pad - 1)
                assert pxs.shape == (2 * pad - 1, 2 * pad - 1)
                arrs['tpi'][y - pad, x - pad] = _Calc.tpi(pxs, nodata)


def _image_add_topography(
    image: numpy.ndarray,
    padded_dem: numpy.ndarray,
    manifest: geo_core.DataBlockManifest,
    features: list[str],
    pad: int,
) -> numpy.ndarray:
    '''Add topographical metrics to the image array.'''
    arrs = {
        'slope': numpy.zeros_like(image[0], dtype=numpy.float32),
        'cos_a': numpy.zeros_like(image[0], dtype=numpy.float32),
        'sin_a': numpy.zeros_like(image[0], dtype=numpy.float32),
        'tpi': numpy.zeros_like(image[0], dtype=numpy.float32),
    }
    max_h, max_w = padded_dem.shape
    if not image[0].shape == (max_h - 2 * pad, max_w - 2 * pad):
        raise ValueError(
            f'Mismatch in image dimensions: {image[0].shape} vs'
            f'({max_h - 2 * pad}, {max_w - 2 * pad}), padding: {pad}'
        )

    _compute_topo_windows(
        padded_dem, arrs, features, pad, manifest['image_nodata']
    )

    band_idx = manifest['image_band_map']
    topos: list[numpy.ndarray] = []
    next_idx = image.shape[0]
    if 'slope' in features:
        topos.extend([arrs['slope'], arrs['cos_a'], arrs['sin_a']])
        band_idx['slope'] = next_idx
        band_idx['cos_a'] = next_idx + 1
        band_idx['sin_a'] = next_idx + 2
        next_idx += 3

    if 'tpi' in features:
        topos.append(arrs['tpi'])
        band_idx['tpi'] = next_idx
        next_idx += 1

    if topos:
        added = numpy.stack(topos, axis=0)
        return numpy.append(image, added, axis=0)
    return image


def _get_image_invalid_mask(
    image: numpy.ndarray,
    nodata: float | None,
) -> numpy.ndarray:
    '''Return True where image values are invalid.'''
    invalid = numpy.isnan(image)
    if nodata is not None:
        if not (isinstance(nodata, float) and numpy.isnan(nodata)):
            invalid |= numpy.isclose(image, nodata)
    return invalid


def _image_get_valid_mask(
    image: numpy.ndarray,
    nodata: float,
) -> tuple[numpy.ndarray, float]:
    '''Get a valid mask and ratio for the whole block.'''
    invalid_img = _get_image_invalid_mask(image, nodata)
    valid_img = ~numpy.any(invalid_img, axis=0)
    ratio = float(numpy.sum(valid_img) / valid_img.size)
    return valid_img, ratio


def _image_get_stats(
    image: numpy.ndarray,
    nodata: float,
) -> dict[str, dict[str, int | float]]:
    '''Per block stats for later aggregation using Welford's.'''
    stats: dict[str, dict[str, int | float]] = {}
    for i, band in enumerate(image):
        mask = _get_image_invalid_mask(band, nodata)
        valid = band[~mask]
        num = valid.size
        if num == 0:
            mean = mean_sq = 0.0
        else:
            mean = numpy.nanmean(valid)
            diff = valid - mean
            mean_sq = numpy.nansum(diff * diff)
            if not numpy.isfinite(mean):
                mean = 0.0
            if not numpy.isfinite(mean_sq):
                mean_sq = 0.0

        stats[f'band_{i}'] = {
            'count': int(num),
            'mean': float(mean),
            'm2': float(mean_sq),
        }
    return stats


def _canonicalize_single_head(
    arr: numpy.ndarray,
    spec: geo_core.CategoricalSpec,
    name: str,
    manifest: geo_core.DataBlockManifest,
) -> numpy.ndarray:
    '''Normalize a single label channel and update manifest statistics.'''
    nodata = manifest['label_nodata']
    ignore_index = manifest['label_ignore_index']
    manifest['label_ignore_cls'][name] = list(spec['ignore_cls'])
    to_ignore = list(spec['ignore_cls']) + [nodata, ignore_index]
    mask = ~numpy.isin(arr, to_ignore)
    shifted_arr = arr + (1 - spec['index_base'])
    normalized = numpy.where(mask, shifted_arr, ignore_index)

    valid = normalized != ignore_index
    ratio = float(valid.sum() / valid.size) if valid.size > 0 else 0.0
    manifest['valid_ratios'][name] = ratio

    n_cls = spec['num_cls']
    manifest['label_num_cls'][name] = n_cls
    valids = normalized[valid].astype(numpy.int64)
    counts = numpy.bincount(valids, minlength=n_cls + 1)[1:n_cls + 1]
    manifest['label_count'][name] = [int(c) for c in counts]

    manifest['label_entropy'][name] = float(_Calc.entropy(counts))

    if 'class_name' in spec and spec['class_name']:
        manifest['label_cls_names'][name] = list(
            spec['class_name'].values()
        )
    if 'color_map' in spec and spec['color_map']:
        manifest['label_cls_clr_map'][name] = spec['color_map']
    if 'taxonomy' in spec and spec['taxonomy']:
        manifest['label_taxonomy'][name] = spec['taxonomy']

    return normalized


def _label_canonicalize(
    raw_label: numpy.ndarray,
    lbl_specs: dict[str, geo_core.CategoricalSpec],
    manifest: geo_core.DataBlockManifest,
) -> numpy.ndarray:
    '''Normalize the label stack based on label specs.'''
    stack: list[numpy.ndarray] = []
    for i, (name, spec) in enumerate(lbl_specs.items(), 1):
        manifest['label_band_map'][name] = i - 1
        arr = raw_label[i - 1]
        normalized = _canonicalize_single_head(
            arr, spec, name, manifest
        )
        stack.append(normalized)

    return numpy.stack(stack, axis=0)


# ----- private classes
class _Calc:
    '''Calculator namespace.'''

    @staticmethod
    def mask(band: numpy.ndarray, nodata: float | None):
        '''Return a masked array where band == nodata.'''
        band_f64 = band.astype(numpy.float64)
        if nodata is None:
            return numpy.ma.array(band_f64, mask=False)
        return numpy.ma.masked_where(
            numpy.isclose(band_f64, nodata), band_f64
        )

    @staticmethod
    def entropy(counts: numpy.ndarray | list[int]) -> float:
        '''Return Shannon entropy.'''
        ent = 0.0
        ss = sum(counts)
        for c in counts:
            if c > 0:
                p = c / ss
                ent -= p * math.log2(p)
        return ent

    @staticmethod
    def ndvi(nir, red, nodata):
        '''Return Normalized Difference Vegetation Index.'''
        out = (nir - red) / (nir + red)
        return out.filled(nodata)

    @staticmethod
    def ndmi(nir, swir1, nodata):
        '''Return Normalized Difference Moisture Index.'''
        out = (nir - swir1) / (nir + swir1)
        return out.filled(nodata)

    @staticmethod
    def nbr(nir, swir2, nodata):
        '''Return Normalized Burn Ratio.'''
        out = (nir - swir2) / (nir + swir2)
        return out.filled(nodata)

    @staticmethod
    def get_px_group(arr: numpy.ndarray, x: int, y: int, rr: int):
        '''Get neighbouring pixels as an array.'''
        return arr[slice(y - rr, y + rr + 1), slice(x - rr, x + rr + 1)]

    @staticmethod
    def slope_n_aspect(arr: numpy.ndarray, nodata: float | None):
        '''Return slope and aspect (in radians) from DEM.'''
        invalid = numpy.isnan(arr).any() or numpy.isinf(arr).any()
        if nodata is not None:
            invalid = invalid or numpy.any(numpy.isclose(arr, nodata))
        if invalid:
            return nodata, nodata, nodata

        dz_dx = (
            (arr[0, 2] + 2 * arr[1, 2] + arr[2, 2]) -
            (arr[0, 0] + 2 * arr[1, 0] + arr[2, 0])
        ) / 8.0
        dz_dy = (
            (arr[2, 0] + 2 * arr[2, 1] + arr[2, 2]) -
            (arr[0, 0] + 2 * arr[0, 1] + arr[0, 2])
        ) / 8.0

        slope = numpy.sqrt(dz_dx ** 2 + dz_dy ** 2)
        if slope == 0.0:
            return 0.0, 1.0, 0.0

        aspect_rad = numpy.arctan2(dz_dy, -dz_dx)
        if aspect_rad < 0:
            aspect_rad += 2 * numpy.pi
        cos_aspect = numpy.cos(aspect_rad)
        sin_aspect = numpy.sin(aspect_rad)
        return slope, cos_aspect, sin_aspect

    @staticmethod
    def tpi(arr: numpy.ndarray, nodata: float | None):
        '''Return Topographical Position Index (TPI) from a DEM.'''
        h, w = arr.shape
        c_row, c_col = h // 2, w // 2
        centre = arr[c_row, c_col]
        if numpy.isnan(centre) or numpy.isinf(centre):
            return nodata
        if nodata is not None and numpy.isclose(centre, nodata):
            return nodata

        invalid_mask = numpy.isnan(arr) | numpy.isinf(arr)
        if nodata is not None:
            invalid_mask |= numpy.isclose(arr, nodata)
        masked = numpy.ma.masked_where(invalid_mask, arr)
        if masked.count() == 1:
            return nodata
        return centre - (masked.sum() - centre) / (masked.count() - 1)
