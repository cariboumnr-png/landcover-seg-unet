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
DataBlock: compile per-block arrays and metadata for geospatial ML.

This module defines a lightweight container and builder for a single
raster *block* (window), designed for geospatial machine learning
pipelines. It operates purely on in-memory NumPy arrays and enriches
them with derived features and structured metadata, without performing
any raster I/O.

Key capabilities:
- Ingest per-window arrays: multi-band image (C, H, W), optional label
  ((1, H, W) → (H, W)), and a padded DEM for neighborhood-based
  topographic analysis.
- Compute derived spectral indices (e.g., NDVI, NDMI, NBR) and
  topographic metrics (slope, aspect components, TPI).
- Construct hierarchical label representations via configurable
  reclassification mappings.
- Generate per-block statistics, including class distributions, Shannon
  entropy, valid pixel ratios, and per-band summary statistics (count,
  mean, M2) suitable for streaming aggregation.
- Serialize and deserialize complete blocks as compressed `.npz`
  artifacts for efficient storage and reproducibility.

Assumptions:
- Input arrays are pre-aligned, windowed, and padded upstream.
- Image arrays follow (C, H, W); labels are (1, H, W) or None.
- DEM padding is sufficient for neighborhood operations.
- Metadata provides required configuration (band mappings, nodata
  values, label schema, etc.).

This module serves as a canonical representation of block-level data,
ensuring consistency and reproducibility across downstream workflows.
'''

# standard imports
from __future__ import annotations
import dataclasses
import json
import math
import typing
# third party imports
import numpy

# ---------------------------------Public Type---------------------------------
class DataBlockMeta(typing.TypedDict):
    '''
    Typed dictionary defining metadata for a single data block.

    This structure captures configuration, provenance, and summary
    statistics associated with a block, covering image, label, and
    derived attributes.

    Categories:
        - General metadata (block identity, validity metrics)
        - Label metadata (schema, hierarchy, distributions)
        - Image metadata (band mapping, nodata handling, statistics)
    '''
    # general metadata
    block_name: str
    has_label: bool
    ignore_index: int
    valid_ratios: dict[str, float]
    # label metadata
    label_nodata: int
    label_num_cls: int
    label_ignore_cls: list[int]
    label_reclass_map: dict[str, list[int]]
    label_ch_parent: dict[str, str | None]
    label_ch_parent_cls: dict[str, int | None]
    label_count: dict[str, list[int]]
    label_entropy: dict[str, float]
    # image metadata
    image_nodata: float
    image_dem_pad: int
    image_band_map: dict[str, int]
    image_stats: dict[str, dict[str, int | float]]

# ------------------------------private dataclass------------------------------
@dataclasses.dataclass
class _BlockArrays:
    '''Simple dataclass for block-wise image/label data.'''
    label: numpy.ndarray = dataclasses.field(init=False)
    label_stack: numpy.ndarray = dataclasses.field(init=False)
    image: numpy.ndarray = dataclasses.field(init=False)
    image_dem_padded: numpy.ndarray = dataclasses.field(init=False)
    valid_mask: numpy.ndarray = dataclasses.field(init=False)

    def validate(self):
        '''Validate if all attr has been populated.'''
        for field in dataclasses.fields(self):
            if not hasattr(self, field.name):
                raise ValueError(f'{field.name} has not been populated yet')

# --------------------------------Public  Class--------------------------------
class DataBlock:
    '''
    Container for per-block raster data and derived metadata.

    A `DataBlock` encapsulates a single raster window along with its
    associated metadata and derived features. It augments raw input
    arrays with spectral indices, topographic metrics, label
    hierarchies, and statistical summaries.

    The class is designed to be I/O-agnostic during construction,
    operating entirely on NumPy arrays provided by upstream processes.
    It supports efficient serialization to and from compressed `.npz`
    artifacts.

    Typical workflow:
        - Use `build()` to construct a block from arrays and metadata
        - Use `save()` to persist the block
        - Use `load()` to restore a previously saved block

    Notes:
        The class implements a staged internal pipeline where image,
        label, and block-level features are computed sequentially.
    '''

    def __init__(self, **kwargs):
        '''
        nitialize an empty `DataBlock` instance.

        The instance is created with placeholder data structures and a
        default metadata dictionary. It can be populated using either
        `build()` (from arrays) or `load()` (from disk).

        Args:
            **kwargs:
                Optional overrides for default metadata values such as:
                - ignore_index: Label ignore value (default: 255)
                - dem_pad: Padding size for DEM-based computations
        '''

        # init with empty block data
        self.data = _BlockArrays()
        # meta dict with default foo values
        self.meta: DataBlockMeta = {
            'block_name': '',
            'has_label': True,
            'ignore_index': kwargs.get('ignore_index', 255),
            'valid_ratios': {},
            'label_nodata': 0,
            'label_num_cls': 0,
            'label_ignore_cls': [],
            'label_reclass_map': {},
            'label_ch_parent': {},
            'label_ch_parent_cls': {},
            'label_count': {},
            'label_entropy': {},
            'image_nodata': numpy.nan,
            'image_dem_pad': kwargs.get('dem_pad', 8),
            'image_band_map': {},
            'image_stats': {}
        }

    # -----------------------------public methods-----------------------------
    @classmethod
    def build(
        cls,
        img_arr: numpy.ndarray,
        lbl_arr: numpy.ndarray | None,
        padded_dem: numpy.ndarray,
        meta: DataBlockMeta,
    ) -> 'DataBlock':
        '''
        Construct a `DataBlock` from in-memory arrays and metadata.

        This method initializes a block, assigns input arrays, and
        executes the full feature engineering pipeline, including:
            - Spectral index computation
            - Topographic metric derivation
            - Label stack construction (if labels are provided)
            - Valid mask generation
            - Per-band statistical summaries

        Args:
            img_arr:
                Image array of shape (C, H, W).
            lbl_arr:
                Optional label array of shape (1, H, W). If None, the
                block is treated as unlabeled.
            padded_dem:
                DEM array padded on all sides to support neighborhood
                operations.
            meta:
                Dictionary containing block configuration and metadata.

        Returns:
            DataBlock:
                A fully populated block instance.

        Notes: The method mutates internal state and returns the instance
        to support chaining.
        '''

        self = cls()
        # update meta with input
        self.meta.update(meta)
        # assign image
        self.data.image = img_arr.astype(numpy.float32) # remote sensing default
        self.data.image_dem_padded = padded_dem.astype(numpy.float32) # padded
        # assign label if provided:
        if lbl_arr is not None:
            # assertions - both 3 dims and the same H and W read by rasterio
            assert len(lbl_arr.shape) == 3 and len(img_arr.shape) == 3
            assert lbl_arr.shape[-2] == img_arr.shape[-2]
            # (1, H, W) -> (H, W)
            self.data.label = lbl_arr.astype(numpy.uint8)[0]
            self.meta['has_label'] = True
        # otherwise give label related data a place holder
        else:
            self.data.label = numpy.array([1])
            self.data.label_stack = numpy.array([1])
            self.meta['has_label'] = False

        # process data sequence
        # image bands related
        self._add_spectral_indices()
        self._add_topographical_metrics()
        # labels related - only if labels are provided
        if self.meta['has_label']:
            self._build_label_stack()
            self._get_topology()
            self._count_label_classes()
        # block-wise
        self._get_block_valid_mask()
        self._get_block_image_stats()

        # sanity check self.data and return self to allow chained calls
        self.data.validate()
        return self

    @classmethod
    def load(cls, fpath: str) -> 'DataBlock':
        '''
        Load a `DataBlock` from a serialized `.npz` file.

        This method reconstructs both the data arrays and metadata
        from a previously saved block artifact.

        Args:
            fpath:
                Path to the `.npz` file containing serialized block data.

        Returns:
            DataBlock:
                A populated block instance with restored state.

        Notes:
            Unknown fields in the archive are ignored during loading.
        '''

        self = cls()
        # load npz file
        loaded = numpy.load(fpath)
        # populate self.data
        for key in loaded:
            if key == 'meta_json':
                continue
            try:
                setattr(self.data, key, loaded[key])
            except AttributeError:
                continue
        # populate self.meta
        self.meta = json.loads(loaded['meta_json'].item())
        # return self to allow chained calls
        return self

    def save(self, fpath: str) -> None:
        '''
        Save the `DataBlock` to a compressed `.npz` file.

        The method serializes all internal arrays along with metadata
        (stored as a compact JSON string) into a single artifact.

        Args:
            fpath:
                Output file path. Must end with `.npz`. Existing files
                will be overwritten.

        Notes:
            Metadata is serialized using JSON to ensure portability.
        '''

        # sanity check
        assert fpath.endswith('.npz')
        # convert self.data dataclass to dict
        to_save = vars(self.data).copy()
        # add meta dict (json dumps to compact plain text)
        meta_json = json.dumps(self.meta, separators=(',', ':'))
        to_save.update({'meta_json': meta_json})
        # save file - allow pickle to write meta dict
        numpy.savez_compressed(fpath, **to_save)

    # -----------------------------internal method-----------------------------
    def _add_spectral_indices(self) -> None:
        '''Add spectral indices using loaded Landsat bands.'''

        # retrieve from meta
        band_indices = self.meta['image_band_map']
        nodata = self.meta['image_nodata']

        # assertions
        assert all(k in band_indices for k in ['red', 'nir', 'swir1', 'swir2'])

        # mask off nodata pixels to avoid overflow
        # 32 bit increased to 64 bit float
        red = _Calc.mask(self.data.image[band_indices['red']], nodata)
        nir = _Calc.mask(self.data.image[band_indices['nir']], nodata)
        swir1 = _Calc.mask(self.data.image[band_indices['swir1']], nodata)
        swir2 = _Calc.mask(self.data.image[band_indices['swir2']], nodata)

        # add to image array
        add_indices = numpy.stack([
            _Calc.ndvi(nir, red, nodata),
            _Calc.ndmi(nir, swir1, nodata),
            _Calc.nbr(nir, swir2, nodata)
        ]).astype(numpy.float32)
        self.data.image = numpy.append(self.data.image, add_indices, axis=0)

        # add to band map
        n = len(self.meta['image_band_map'])
        self.meta['image_band_map'].update({
            'NDVI': n,
            'NDMI': n + 1,
            'NBR': n + 2
        })

    def _add_topographical_metrics(self) -> None:
        '''Add topographical metrics to the image array.'''

        # get vars
        nodata = self.meta['image_nodata']

        # prep metrics to add
        slope = numpy.zeros_like(self.data.image[0], dtype=numpy.float32)
        cos_a = numpy.zeros_like(self.data.image[0], dtype=numpy.float32)
        sin_a = numpy.zeros_like(self.data.image[0], dtype=numpy.float32)
        tpi = numpy.zeros_like(self.data.image[0], dtype=numpy.float32)

        # padding
        pad = self.meta['image_dem_pad']
        # sanity check on image/padded image shape
        max_h, max_w = self.data.image_dem_padded.shape
        if not self.data.image[0].shape == (max_h - 2 * pad, max_w - 2 * pad):
            raise ValueError(f'{self.data.image[0].shape} {max_h}, {max_w}, {pad}')

        # iterate through pixels from the original block in padded dem
        for y in range(pad, max_h - pad):
            for x in range(pad, max_w - pad):
                # slope and aspect - pad neighbors, radius 1
                pxs = _Calc.get_px_group(self.data.image_dem_padded, x, y, 1)
                slope[y - pad, x - pad], cos_a[y - pad, x - pad], \
                    sin_a[y - pad, x - pad] = _Calc.slope_n_aspect(pxs, nodata)
                # tpi - 224 neighbors, radius pad-1
                r = pad - 1
                pxs =  _Calc.get_px_group(self.data.image_dem_padded, x, y, r)
                tpi[y - pad, x - pad] = _Calc.tpi(pxs, nodata)

        # add to image array
        to_add = numpy.stack([slope, cos_a, sin_a, tpi], axis=0)
        self.data.image = numpy.append(self.data.image, to_add, axis=0)

        # add to band map
        n = len(self.meta['image_band_map'])
        self.meta['image_band_map'].update({
            'slope': n,
            'cos_aspect': n + 1,
            'sin_aspect': n + 2,
            'tpi': n + 3
        })

    def _build_label_stack(self) -> None:
        '''Build the label stack and update its valid pixel ratios.'''

        # get values from meta
        ignore_index = self.meta['ignore_index']
        reclass_map = self.meta['label_reclass_map']

        # get labels to be ignored
        labels_to_ignore = list(self.meta['label_ignore_cls'])
        labels_to_ignore.append(self.meta['label_nodata'])
        labels_to_ignore = [x for x in labels_to_ignore if x is not None]

        # base as the first element of the list
        raw_mask = ~numpy.isin(self.data.label, labels_to_ignore)
        raw_valid = numpy.where(raw_mask, self.data.label, ignore_index)
        fn_stack = [raw_valid]
        self.meta['valid_ratios'].update({
            'base': float(numpy.sum(raw_mask) / raw_valid.size)
        })

        # if no reclass, assign the only base label channel to self.data
        if not reclass_map:
            self.data.label_stack = numpy.stack(fn_stack, axis=0)
            return

        # iterate through reclass map
        for band_num, classes in reclass_map.items():
            # mask to the current base channel class IDs (parent)
            _mask = numpy.isin(self.data.label, classes)
            # in-place reclass relevant pixels in layer1
            fn_stack[0][_mask] = int(band_num)
            # create a new array to add to stack
            reclass_new = numpy.where(_mask, self.data.label, ignore_index)
            # reclass from 1 to n
            for i, cls in enumerate(classes, 1):
                reclass_new[reclass_new == cls] = int(i)
            # append the to the stack
            fn_stack.append(reclass_new)
        # stack all the arrays and assign to self.data
        self.data.label_stack = numpy.stack(fn_stack, axis=0)

        # update valid pixel ratios for the stack
        for i in range(1, len(reclass_map) + 1):
            _valid = fn_stack[i] != ignore_index
            _ratio = float(numpy.sum(_valid) / raw_valid.size)
            self.meta['valid_ratios'].update({f'reclass_{i}': _ratio})

    def _get_topology(self) -> None:
        '''Derive label topology (parent-child).'''

        # iterate through label counts
        for layer_name in self.meta['valid_ratios']:
            # emit topology
            if layer_name == 'base':
                self.meta['label_ch_parent'][layer_name] = None
                self.meta['label_ch_parent_cls'][layer_name] = None
            elif layer_name.startswith('reclass_'):
                cls_id = int(layer_name.split('reclass_')[1])
                self.meta['label_ch_parent'][layer_name] = 'base'
                self.meta['label_ch_parent_cls'][layer_name] = cls_id
            else:
                pass

    def _count_label_classes(self) -> None:
        '''Count present label values and calculate entropy.'''

        # supposed number of classes for each label layer
        label_num = self.meta['label_num_cls']
        reclass_map = self.meta['label_reclass_map']

        # when no reclass, treat layer1 as original classes
        if not reclass_map:
            n_classes = [label_num, label_num]
        # count of original label and reclassed layer1 groups
        else:
            n_classes = [label_num, len(reclass_map)]
            # counts of layer2 groups
            n_classes.extend([len(v) for v in reclass_map.values()])

        # all arrays to be counted
        channels = numpy.concatenate(
            [self.data.label[None, :, :], self.data.label_stack], axis=0
        ) # (L, H, W)

        # iterate label layers
        assert len(n_classes) == len(channels) # sanity check
        for i, band in enumerate(channels):
            # count unique values for each label layer
            label_unique = numpy.arange(1, n_classes[i] + 1) # start from 1
            filtered = band[numpy.isin(band, label_unique)]
            uniques, counts = numpy.unique(filtered, return_counts=True)

            # convert to list. avoid using arr.tolist()
            uniques = [int(_) for _ in uniques]
            counts = [int(_) for _ in counts]

            # get shannon entropy
            ent = _Calc.entropy(counts)

            # assign zero count to no_show classes
            cc = []
            for _ in range(n_classes[i]):
                idx = _ + 1
                if idx in uniques:
                    count_idx = uniques.index(idx)
                    cc.append(counts[count_idx])
                else:
                    cc.append(0)

            # add to metadata
            if i == 0:
                self.meta['label_count'].update({'original': cc})
                self.meta['label_entropy'].update({'original': ent})
            elif i == 1:
                self.meta['label_count'].update({'base': cc})
                self.meta['label_entropy'].update({'base': ent})
            else:
                self.meta['label_count'].update({f'reclass_{i - 1}': cc})
                self.meta['label_entropy'].update({f'reclass_{i - 1}': ent})

    def _get_block_valid_mask(self):
        '''Get a valid mask for the whole block.'''

        # get image nodata and ignore label index
        ignore_index = self.meta['ignore_index']
        image_nodata = self.meta['image_nodata']

        # for label data: if provided, True where label layer1 is valid
        if self.meta['has_label']:
            valid_label = self.data.label_stack[0] != ignore_index
        # otherwise no effect
        else:
            valid_label = True # easy broadcast

        # for image data: True where all image bands are valid
        invalid_img = (
            numpy.isnan(self.data.image) |
            numpy.isclose(self.data.image, image_nodata)
        )
        valid_img = ~numpy.any(invalid_img, axis=0) # shape (256, 256)

        # final mask as combined
        self.data.valid_mask = valid_label & valid_img # (256, 256)

        # add to meta
        self.meta['valid_ratios'].update({
            'image': float((numpy.sum(valid_img) / valid_img.size)),
            'block': float((numpy.sum(self.data.valid_mask) / valid_img.size))
        })

    def _get_block_image_stats(self):
        '''Per block stats for later aggregation using Welford's.'''

        # image_nodata
        image_nodata = self.meta['image_nodata']
        # iterate through image channels
        for i, band in enumerate(self.data.image):
            # get where pixel is invalid and inverse to get valid pixels
            if isinstance(image_nodata, float) and numpy.isnan(image_nodata):
                mask = numpy.isnan(band)
            else:
                mask = numpy.isclose(band, image_nodata)
            valid = band[~mask]

            num = valid.size
            if num == 0:
                mean = mean_sq = 0.0 # safe neutral values
            else:
                # nan-safe ops in case stray NaNs remain
                mean = numpy.nanmean(valid)
                diff = valid - mean
                mean_sq = numpy.nansum(diff * diff)
                # final guard against numerical weirdness
                if not numpy.isfinite(mean):
                    mean = 0.0
                if not numpy.isfinite(mean_sq):
                    mean_sq = 0.0

            # give to self.image_stats
            self.meta['image_stats'][f'band_{i}'] = {
                'count': int(num), 'mean': float(mean), 'm2': float(mean_sq)
            }

# --------------------------------private class--------------------------------
class _Calc:
    '''Calculator namespace.'''
    # spectral indices related
    @staticmethod
    def mask(band, nodata):
        '''Returns a mask where band == nodata.'''
        # avoid overflow downstream
        return numpy.ma.masked_where(
            numpy.isclose(band, nodata), band.astype(numpy.float64)
        )

    @staticmethod
    def entropy(counts):
        '''Shannon entropy.'''
        ent = 0.0
        ss = sum(counts)
        for c in counts:
            if c > 0:
                p = c / ss
                ent -= p * math.log2(p)
        return ent

    @staticmethod
    def ndvi(nir, red, nodata):
        '''Returns NDVI.'''
        out = (nir - red) / (nir + red)
        return out.filled(nodata)

    @staticmethod
    def ndmi(nir, swir1, nodata):
        '''Returns NDMI'''
        out = (nir - swir1) / (nir + swir1)
        return out.filled(nodata)

    @staticmethod
    def nbr(nir, swir2, nodata):
        '''Returns Normalized Burn Ratio (NBR).'''
        out = (nir - swir2) / (nir + swir2)
        return out.filled(nodata)

    # topographical metrics related
    @staticmethod
    def get_px_group(arr, x, y, np):
        '''Get neighbouring pixels as an array.'''
        return arr[slice(y - np, y + np + 1), slice(x - np, x + np + 1)]

    @staticmethod
    def slope_n_aspect(arr, nodata):
        '''Returns slope and aspect from DEM.'''
        # all 9 cells need to have a valid value (Horn's)
        if numpy.any(numpy.isclose(arr, nodata)) or \
            numpy.isnan(arr).any() or \
                numpy.isinf(arr).any():
            return nodata, nodata, nodata
        # calculation
        dz_dx = (
            (arr[0, 2] + 2 * arr[1, 2] + arr[2, 2]) - \
            (arr[0, 0] + 2 * arr[1, 0] + arr[2, 0])
        ) / 8.0
        dz_dy = (
            (arr[2, 0] + 2 * arr[2, 1] + arr[2, 2]) - \
            (arr[0, 0] + 2 * arr[0, 1] + arr[0, 2])
        ) / 8.0
        # calculate slope
        slope = numpy.sqrt(dz_dx ** 2 + dz_dy ** 2)
        # calculate aspect angle in radians
        aspect_rad = numpy.arctan2(dz_dy, -dz_dx)
        if aspect_rad < 0:
            aspect_rad += 2 * numpy.pi  # Normalize to [0, 2π]
        # compute cosine and sine of aspect
        cos_aspect = numpy.cos(aspect_rad)
        sin_aspect = numpy.sin(aspect_rad)
        # return
        return slope, cos_aspect, sin_aspect

    @staticmethod
    def tpi(arr, nodata):
        '''Returns Topographical Position Index (TPI) from DEM.'''
        # topographical position index
        h, w = arr.shape
        c_row, c_col = h // 2, w // 2
        centre = arr[c_row, c_col]
        # centre pixel is nodata
        if numpy.isclose(centre, nodata):
            return nodata
        masked = numpy.ma.masked_where(numpy.isclose(arr, nodata), arr)
        # all is nodata except centre
        if masked.count() == 1:
            return nodata
        # valid arr
        return centre - (masked.sum() - centre) / (masked.count() - 1)
