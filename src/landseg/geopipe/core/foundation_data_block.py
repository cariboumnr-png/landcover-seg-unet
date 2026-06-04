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
    label_num_cls: dict[str, int]
    label_ignore_cls: dict[str, list[int]]
    label_parent: dict[str, str | None]
    label_parent_cls: dict[str, int | None]
    label_count: dict[str, list[int]]
    label_entropy: dict[str, float]
    label_names: dict[str, list[str]]
    # image metadata
    image_nodata: float
    image_dem_pad: int
    image_band_map: dict[str, int]
    image_stats: dict[str, dict[str, int | float]]

class LabelSpecs(typing.TypedDict):
    '''Typed dictionary for label specification.'''
    # required
    num_cls: int
    ignore_cls: list[int]
    # optional
    class_name: typing.NotRequired[dict[str, str]]
    reclass: typing.NotRequired[dict[str, list[int]]]
    reclass_name: typing.NotRequired[dict[str, str]]

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

    def __init__(
        self,
        *,
        ignore_index: int = 255,
        dem_pad: int = 8,
    ):
        '''
        Initialize an empty `DataBlock` instance.

        The instance is created with placeholder data structures and a
        default metadata dictionary. It can be populated using either
        `build()` (from arrays) or `load()` (from disk).

        Args:
            ignore_index: Label ignore value (default: 255)
            dem_pad: Padding size for DEM-based computations (default: 8)
        '''

        # init with empty block data
        self.data = _BlockArrays()
        # meta dict with default foo values
        self.meta: DataBlockMeta = {
            'block_name': '',
            'has_label': True,
            'ignore_index': ignore_index,
            'valid_ratios': {},
            'label_nodata': 0,
            'label_num_cls': {},
            'label_ignore_cls': {},
            'label_parent': {},
            'label_parent_cls': {},
            'label_count': {},
            'label_entropy': {},
            'label_names': {},
            'image_nodata': numpy.nan,
            'image_dem_pad': dem_pad,
            'image_band_map': {},
            'image_stats': {}
        }

    # ----- alternative constructor
    @classmethod
    def build(
        cls,
        *,
        raw_arrays: tuple[numpy.ndarray, numpy.ndarray | None],
        img_padded_dem: numpy.ndarray,
        lbl_specs: dict[str, LabelSpecs],
        block_meta: DataBlockMeta,
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
            img_arr: Image array of shape (C, H, W).
            lbl_arr: Optional label array of shape (T, H, W). If None,
                the block is treated as unlabeled.
            padded_dem: DEM array padded on all sides to support
                neighborhood operations.
            meta: Dictionary with block configuration and metadata.

        Returns:
            DataBlock: A fully populated block instance.

        Notes: The method mutates internal state and returns the instance
        to support chaining.
        '''

        self = cls()
        img_arr, lbl_arr = raw_arrays
        # update meta with input
        self.meta.update(block_meta)
        # assign image
        self.data.image = img_arr.astype(numpy.float32) # remote sensing default
        self.data.image_dem_padded = img_padded_dem.astype(numpy.float32) # padded
        # assign label if provided:
        if lbl_arr is not None:
            # assertions - both 3 dims with the same H and W
            assert len(lbl_arr.shape) == 3 and len(img_arr.shape) == 3
            assert lbl_arr.shape[-2] == img_arr.shape[-2]
            self.data.label_stack = lbl_arr.astype(numpy.uint8)
            self.data.label = self.data.label_stack[0]
            self.meta['has_label'] = True
        # otherwise give label related data a place holder
        else:
            self.data.label = numpy.array([1])
            self.data.label_stack = numpy.array([1])
            self.meta['has_label'] = False

        # process data sequence
        # --- add image bands
        self._image_add_spectral()
        self._image_add_topography()

        # --- image specs
        self._image_get_valid_mask()
        self._image_get_stats()

        # ---labels specs
        # only if labels are provided
        if self.meta['has_label']:
            self._label_get_stack(lbl_specs)
            self._label_get_stats()

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
            fpath: Path to the `.npz` file containing serialized data.

        Returns:
            DataBlock: A populated block instance with restored state.

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

    # ----- public method
    def save(self, fpath: str) -> None:
        '''
        Save the `DataBlock` to a compressed `.npz` file.

        The method serializes all internal arrays along with metadata
        (stored as a compact JSON string) into a single artifact.

        Args:
            fpath: Output file path. Must end with `.npz`. Existing files
                **will** be overwritten.

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

    # ----- private method
    def _image_add_spectral(self) -> None:
        '''Add spectral indices if related bands are available.'''

        # retrieve from meta
        band_idx = self.meta['image_band_map']
        nodata = self.meta['image_nodata']

        # mask off nodata pixels to avoid overflow
        # 32 bit increased to 64 bit float
        red = _Calc.mask(self.data.image[band_idx['red']], nodata)
        # add spectral indices depending on band availability
        indices_stack = []
        n = len(self.meta['image_band_map'])
        if 'nir' in band_idx:
            nir = _Calc.mask(self.data.image[band_idx['nir']], nodata)
            indices_stack.append(_Calc.ndvi(nir, red, nodata))
            self.meta['image_band_map']['NDVI'] = n
            if 'swir1' in band_idx:
                swir1 = _Calc.mask(self.data.image[band_idx['swir1']], nodata)
                indices_stack.append(_Calc.ndmi(nir, swir1, nodata))
                self.meta['image_band_map']['NDMI'] = n + 1
            if 'swir2' in band_idx:
                swir2 = _Calc.mask(self.data.image[band_idx['swir2']], nodata)
                indices_stack.append(_Calc.nbr(nir, swir2, nodata))
                self.meta['image_band_map']['NBR'] = n + 2

        # add to image array
        if indices_stack:
            added = numpy.stack(indices_stack ).astype(numpy.float32)
            self.data.image = numpy.append(self.data.image, added, axis=0)

    def _image_add_topography(self) -> None:
        '''Add topographical metrics to the image array.'''

        # retrieve from meta
        nodata = self.meta['image_nodata']
        pad = self.meta['image_dem_pad']

        # prep metrics to add
        slope = numpy.zeros_like(self.data.image[0], dtype=numpy.float32)
        cos_a = numpy.zeros_like(self.data.image[0], dtype=numpy.float32)
        sin_a = numpy.zeros_like(self.data.image[0], dtype=numpy.float32)
        tpi = numpy.zeros_like(self.data.image[0], dtype=numpy.float32)

        # sanity check on image/padded image shape
        max_h, max_w = self.data.image_dem_padded.shape
        if not self.data.image[0].shape == (max_h - 2 * pad, max_w - 2 * pad):
            raise ValueError(
                f'Mismatch in image dimensions: {self.data.image[0].shape} vs'
                f'({max_h - 2 * pad}, {max_w - 2 * pad}), padding: {pad}'
            )

        # iterate through pixels from the original block in padded dem
        for y in range(pad, max_h - pad):
            for x in range(pad, max_w - pad):
                # slope and aspect - pad neighbors, radius=1
                pxs = _Calc.get_px_group(self.data.image_dem_padded, x, y, 1)
                slope[y - pad, x - pad], cos_a[y - pad, x - pad], \
                    sin_a[y - pad, x - pad] = _Calc.slope_n_aspect(pxs, nodata)
                # tpi - radius=pad-1 (default 8, so 15x15 window)
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

    def _image_get_valid_mask(self):
        '''Get a valid mask for the whole block.'''

        # for image data: True where all image bands are valid
        invalid_img = (
            numpy.isnan(self.data.image) |
            numpy.isclose(self.data.image, self.meta['image_nodata'])
        )
        valid_img = ~numpy.any(invalid_img, axis=0) # shape (256, 256)

        self.meta['valid_ratios'].update({
            'image': float((numpy.sum(valid_img) / valid_img.size)),
        })
        self.data.valid_mask = valid_img # (256, 256)

    def _image_get_stats(self):
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

    def _label_get_stack(self, lbl_specs: dict[str, LabelSpecs]):
        '''
        Construct a multi-layer label stack based on specs and reclass rules.

        Input arrays are processed sequentially. Bands without reclass maps are
        added as-is. Bands with reclass maps are converted into group-ID layers,
        followed by additional slices for each group (child layers).
        '''

        # containers
        num_cls: dict[str, int] = {}
        ignore_cls: dict[str, list[int]] = {}
        parent_map: dict[str, str | None] = {}
        parent_cls_map: dict[str, int | None] = {}
        label_names: dict[str, list[str]] = {}

        stack: list[numpy.ndarray] = []
        # iterate label specs
        ignore_index = self.meta['ignore_index']
        for i, (name, spec) in enumerate(lbl_specs.items()):

            arr = self.data.label[i]
            reclass_name = spec.get('reclass_name', {})
            cls_name = spec.get('class_name', {})

            # 1. Add the Raw Masked band (Original Class IDs)
            to_ignore = list(spec['ignore_cls']) + [ignore_index]
            raw_mask = ~numpy.isin(arr, to_ignore)
            raw_valid = numpy.where(raw_mask, arr, ignore_index)
            stack.append(raw_valid)

            label_names[name] = [
                cls_name.get(str(j + 1), f'cls_{j + 1}')
                for j in range(spec['num_cls'])
            ]

            num_cls[name] = spec['num_cls']
            ignore_cls[name] = spec['ignore_cls']
            parent_map[name] = None
            parent_cls_map[name] = None

            reclass = spec.get('reclass')
            if not reclass:
                continue

            # 2. Add a Grouping Band (Group IDs)
            # This band becomes the "parent" for the child slices
            grp_name = f'{name}_groups'
            group_layer = numpy.full_like(arr, ignore_index, dtype=arr.dtype)

            # Populate Grouping band and create Children
            for group_id, classes in reclass.items():
                _mask = numpy.isin(arr, classes)
                group_layer[_mask] = int(group_id)

                # 3. Create Child slices
                child_arr = numpy.where(_mask, arr, ignore_index)
                # re-index original classes in this slice to 1..N
                for k, cls in enumerate(classes, 1):
                    child_arr[child_arr == cls] = int(k)

                # Append child to stack after grouping layer is pushed
                # (We'll handle the stack push order below)
                child_name = reclass_name.get(group_id, f'{grp_name}_{group_id}')

                # Meta for child (parent is the grouping layer)
                num_cls[child_name] = len(classes)
                ignore_cls[child_name] = []
                parent_map[child_name] = grp_name
                parent_cls_map[child_name] = int(group_id)

                # Child class names (derived from original class names)
                label_names[child_name] = [
                    cls_name.get(str(c), f'cls_{c}')
                    for c in classes
                ]

                # Temporary list to manage push order if needed,
                # but here we just append to the global stack
                stack.append(child_arr)

            # Insert/Append Group metadata and data
            # To keep it logical, we add the grouping band to the stack
            # but we need to track where it is. Let's append it at the end
            # of the "block" for this spec.
            stack.append(group_layer)
            num_cls[grp_name] = len(reclass)
            ignore_cls[grp_name] = []
            parent_map[grp_name] = None
            parent_cls_map[grp_name] = None

            # Group names (sorted by group ID to match class indices 1..N)
            sorted_gids = sorted(reclass.keys(), key=int)
            label_names[grp_name] = [
                reclass_name.get(gid, f'grp_{gid}')
                for gid in sorted_gids
            ]

        # stack all the arrays
        self.data.label_stack = numpy.stack(stack, axis=0)
        # populate meta dict
        self.meta['label_num_cls'] = num_cls
        self.meta['label_ignore_cls'] = ignore_cls
        self.meta['label_parent'] = parent_map
        self.meta['label_parent_cls'] = parent_cls_map
        self.meta['label_names'] = label_names

    def _label_get_stats(self) -> None:
        '''Count present label values and calculate entropy.'''

        # the meta dict defines the names and sizes of every target in stack
        heads = list(self.meta['label_num_cls'].items())

        for i, (name, n_cls) in enumerate(heads):
            band = self.data.label_stack[i]

            # calculate valid pixel ratios
            valid = band != self.meta['ignore_index']
            self.meta['valid_ratios'].update({
                name: float(valid.sum() / (valid.size))
                if valid.size > 0 else 0.0
            })

            # count unique values for the current head (classes 1..N)
            label_unique = numpy.arange(1, n_cls + 1)
            filtered = band[numpy.isin(band, label_unique)]
            uniques, counts = numpy.unique(filtered, return_counts=True)

            # calculate shannon entropy
            ent = float(_Calc.entropy(counts))

            # align counts to the fixed class size defined in schema
            final_counts = [0] * n_cls
            for val, count in zip(uniques, counts):
                final_counts[int(val) - 1] = int(count)

            # store results in meta
            self.meta['label_count'][name] = final_counts
            self.meta['label_entropy'][name] = ent

# --------------------------------private class--------------------------------
class _Calc:
    '''Calculator namespace.'''
    @staticmethod
    def mask(band, nodata):
        '''Returns a masked array where band == nodata.'''
        band = band.astype(numpy.float64)
        if nodata is None: # if nodata is None, no values are masked.
            return numpy.ma.array(band, mask=False)
        return numpy.ma.masked_where(numpy.isclose(band, nodata), band)

    @staticmethod
    def entropy(counts):
        '''Returns Shannon entropy.'''
        ent = 0.0
        ss = sum(counts)
        for c in counts:
            if c > 0:
                p = c / ss
                ent -= p * math.log2(p)
        return ent

    @staticmethod
    def ndvi(nir, red, nodata):
        '''Returns Normalized Difference Vegetation Index.'''
        out = (nir - red) / (nir + red)
        return out.filled(nodata)

    @staticmethod
    def ndmi(nir, swir1, nodata):
        '''Returns Normalized Difference Moisture Index.'''
        out = (nir - swir1) / (nir + swir1)
        return out.filled(nodata)

    @staticmethod
    def nbr(nir, swir2, nodata):
        '''Returns Normalized Burn Ratio.'''
        out = (nir - swir2) / (nir + swir2)
        return out.filled(nodata)

    # topographical metrics related
    @staticmethod
    def get_px_group(arr, x, y, np):
        '''Get neighbouring pixels as an array.'''
        return arr[slice(y - np, y + np + 1), slice(x - np, x + np + 1)]

    @staticmethod
    def slope_n_aspect(arr, nodata):
        '''Returns slope and aspect (in radians) from DEM.'''
        # all 9 cells need to have a valid value (Horn's)
        invalid = numpy.isnan(arr).any() or numpy.isinf(arr).any()
        if nodata is not None:
            invalid = invalid or numpy.any(numpy.isclose(arr, nodata))
        if invalid:
            return nodata, nodata, nodata
        # calculation
        dz_dx = (
            (arr[0, 2] + 2 * arr[1, 2] + arr[2, 2]) -
            (arr[0, 0] + 2 * arr[1, 0] + arr[2, 0])
        ) / 8.0
        dz_dy = (
            (arr[2, 0] + 2 * arr[2, 1] + arr[2, 2]) -
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
        return slope, cos_aspect, sin_aspect

    @staticmethod
    def tpi(arr, nodata):
        '''Returns Topographical Position Indexfrom DEM.'''
        # topographical position index
        h, w = arr.shape
        c_row, c_col = h // 2, w // 2
        centre = arr[c_row, c_col]
        # invalid centre pixel
        if numpy.isnan(centre) or numpy.isinf(centre):
            return nodata
        if nodata is not None and numpy.isclose(centre, nodata):
            return nodata
        # build mask
        invalid_mask = numpy.isnan(arr) | numpy.isinf(arr)
        if nodata is not None:
            invalid_mask |= numpy.isclose(arr, nodata)
        masked = numpy.ma.masked_where(invalid_mask, arr)
        # all is nodata except centre
        if masked.count() == 1:
            return nodata
        # valid arr
        return centre - (masked.sum() - centre) / (masked.count() - 1)
