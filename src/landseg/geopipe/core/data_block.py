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
DataBlock: compile per-block arrays and manifest for geospatial ML.

This module defines a lightweight container and builder for a single
raster *block* (window), designed for geospatial machine learning
pipelines. It operates purely on in-memory NumPy arrays and enriches
them with derived features and structured manifest, without performing
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
- Build configuration provides feature engineering options, while the
  persisted manifest records the dataset schema, provenance, and
  per-block statistics.

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
class DataBlockManifest(typing.TypedDict):
    '''
    Typed dictionary defining the persisted manifest for a data block.

    The manifest records block provenance, dataset schema, and derived
    statistics required to interpret the serialized arrays independently
    of the original dataset configuration.
    '''
    # provenance
    block_name: str
    has_label: bool
    # image description
    image_band_map: dict[str, int]
    image_nodata: float
    # label descriptions
    label_band_map: dict[str, int] # head name: index
    label_nodata: int
    label_ignore_index: int # ignore_idx to convert to globally
    label_ignore_cls: dict[str, list[int]] # head name: list of ignores
    label_num_cls: dict[str, int]
    label_cls_names: dict[str, list[str]] # head name: class names as a list
    label_cls_clr_map: dict[str, dict[str, list[int]]] # head name: {class name: RGB color}

    # NOTE: to migrate to data prep
    # label_parent: dict[str, str | None]
    # label_parent_cls: dict[str, int | None]
    # label_taxonomy: dict[str, typing.Any]

    # derived stats
    valid_ratios: dict[str, float]
    image_stats: dict[str, dict[str, int | float]]
    label_count: dict[str, list[int]]
    label_entropy: dict[str, float]


class LabelSpecs(typing.TypedDict):
    '''Typed dictionary for label specification.'''
    # required
    num_cls: int
    ignore_cls: list[int]
    index_base: int
    # optional
    class_name: typing.NotRequired[dict[str, str]]
    color_map: typing.NotRequired[dict[str, list[int]]]

    # NOTE: to migrate to data prep
    # reclass: typing.NotRequired[dict[str, list[int]]]
    # reclass_name: typing.NotRequired[dict[str, str]]
    # taxonomy: typing.NotRequired[dict[str, typing.Any]]


# ------------------------------Public  Dataclass------------------------------
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
            # if self.label_specs is None:
            #     raise ValueError('Label array present but specs not provided')
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

    label_nodata: int = 0
    label_ignore_index: int = 255
    label_band_map: dict[str, int] = dataclasses.field(default_factory=dict)
    label_specs: dict[str, LabelSpecs] = dataclasses.field(default_factory=dict)

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
            if 'ndvi' in spectral and not 'nir' in band_map:
                raise ValueError('NDVI calculation: NIR band missing')
            if 'ndmi' in spectral and not 'swir1' in band_map:
                raise ValueError('NDMI calculation: SWIR1 band missing')
            if 'nbr' in spectral and not 'swir2' in band_map:
                raise ValueError('NBR calculation: SWIR2 band missing')

        if self.add_topo:
            topo  = [t.lower() for t in self.add_topo]
            invalid = [t for t in topo if t not in ['slope', 'aspect', 'tpi']]
            if invalid:
                raise ValueError(f'Invalid topo features: {invalid}')

            if 'dem' not in band_map:
                raise ValueError('DEM band missing for topographical features')

# ------------------------------private dataclass------------------------------
@dataclasses.dataclass
class _BlockArrays:
    '''Simple dataclass for block-wise image/label data.'''
    label: numpy.ndarray = dataclasses.field(init=False)
    image: numpy.ndarray = dataclasses.field(init=False)
    valid_mask: numpy.ndarray = dataclasses.field(init=False)

    def validate(self):
        '''Validate if all attributes have been populated.'''
        required = ('image', 'valid_mask')
        for name in required:
            if not hasattr(self, name):
                raise ValueError(f'{name} has not been populated yet')


# --------------------------------Public  Class--------------------------------
class DataBlock:
    '''
    Container for per-block raster data and its associated manifest.

    A `DataBlock` encapsulates a single raster window along with its
    associated manifest and derived features. It augments raw input
    arrays with spectral indices, topographic metrics, label
    hierarchies, and statistical summaries.

    The class is designed to be I/O-agnostic during construction,
    operating entirely on NumPy arrays provided by upstream processes.
    It supports efficient serialization to and from compressed `.npz`
    artifacts.

    Typical workflow:
        - Use `build()` to construct a block from arrays and manifest
        - Use `save()` to persist the block
        - Use `load()` to restore a previously saved block

    Notes:
        The class implements a staged internal pipeline where image,
        label, and block-level features are computed sequentially.
    '''

    def __init__(self):
        '''
        Initialize an empty `DataBlock` instance.

        The instance is created with placeholder data structures and a
        default manifest. It can be populated using either `build()`
        (from arrays) or `load()` (from disk).

        Args:
            add_spectral
            add_topo
        '''
        # init shared state/variables for construction
        self.data = _BlockArrays()
        self.padded_dem = numpy.array([1])
        self.manifest: DataBlockManifest = {
            # provenance
            'block_name': '',
            'has_label': False,
            # dataset description
            'image_band_map': {},
            'image_nodata': numpy.nan,
            'label_band_map': {},
            'label_nodata': 0,
            'label_ignore_index': 255,
            'label_ignore_cls': {},
            'label_num_cls': {},
            'label_cls_names': {},
            'label_cls_clr_map': {},
            # derived stats
            'valid_ratios': {},
            'image_stats': {},
            'label_count': {},
            'label_entropy': {},
        }
        # labels speces
        self.lbl_specs: dict[str, LabelSpecs] = {}

    # ----- alternative constructor
    @classmethod
    def build(
        cls,
        inputs: DataBlockInputs,
        config: DataBlockConfig
    ) -> 'DataBlock':
        '''
        Construct a DataBlock from source arrays and build configuration.

        This method initializes a block, assigns input arrays, and
        executes the full feature engineering pipeline, including:
            - Spectral index computation
            - Topographic metric derivation
            - Label stack construction (if labels are provided)
            - Valid mask generation
            - Per-band statistical summaries

        Args:
            inputs:
                Source arrays and dataset schema required to construct
                the block.
            config:
                Build configuration controlling feature engineering and
                data encoding.

        Returns:
            DataBlock: A fully populated block instance.

        Notes: The method mutates internal state and returns the instance
        to support chaining.
        '''
        self = cls()

        # update manifest dict
        self.manifest.update({
            'block_name': inputs.block_name,
            'image_nodata': config.image_nodata,
            'image_band_map': dict(config.image_band_map), # shallow copy
        })

        # image dtype conversion and processing
        # float32 as remote sensing default
        self.data.image = inputs.image_array.astype(numpy.float32)

        if config.add_spectral:
            self._image_add_spectral(
                [item.lower() for item in config.add_spectral or []]
            )

        if config.add_topo:
            self.padded_dem = inputs.pad_dem.astype(numpy.float32)
            self._image_add_topography(
                [item.lower() for item in config.add_topo or []],
                config.image_dem_pad_px
            )

        self._image_get_valid_mask()
        self._image_get_stats()

        # if label array is provided:
        if inputs.has_label:

            if not config.label_band_map:
                raise ValueError('"label_band_map" must not provide')
            if not config.label_specs:
                raise ValueError('"label_specs" not provided')

            self.manifest.update({
                'label_nodata': config.label_nodata,
                'label_band_map': dict(config.label_band_map),
                'label_ignore_index': config.label_ignore_index,
            })

            # currently support labels range [0, 256)
            assert inputs.label_array is not None
            self.data.label = inputs.label_array.astype(numpy.uint8)
            self.lbl_specs = dict(config.label_specs)
            self._label_canonicalize()
            self.manifest['has_label'] = True

        else:
            self.data.label = numpy.array([1]) # dummy placeholders
            self.manifest['has_label'] = False

        # sanity check self.data and return self to allow chained calls
        self.data.validate()
        return self

    @classmethod
    def load(cls, fpath: str) -> 'DataBlock':
        '''
        Load a `DataBlock` from a serialized `.npz` file.

        This method reconstructs both the data arrays and manifest from
        a previously saved block artifact.

        Args:
            fpath:
                Path to the `.npz` file containing serialized data.

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
            if key == 'manifest_json':
                continue
            try:
                setattr(self.data, key, loaded[key])
            except AttributeError:
                continue

        self.manifest = json.loads(loaded['manifest_json'].item())
        return self # return self to allow chained calls

    # ----- public method
    def save(self, fpath: str) -> None:
        '''
        Save the `DataBlock` to a compressed `.npz` file.

        The method serializes all internal arrays along with manifest
        (stored as a compact JSON string) into a single artifact.

        Args:
            fpath:
                Output file path. Must end with `.npz`. Existing files
                **will** be overwritten.

        Notes:
            Metadata is serialized using JSON to ensure portability.
        '''
        assert fpath.endswith('.npz') # sanity check

        # convert self.data dataclass to dict
        to_save = vars(self.data).copy()
        # add meta dict (json dumps to compact plain text)
        manifest = json.dumps(self.manifest, separators=(',', ':'))
        to_save.update({'manifest_json': manifest})
        # save file - allow pickle to write dict
        numpy.savez_compressed(fpath, **to_save)

    # ----- private method
    def _image_add_spectral(self, indices: list[str]) -> None:
        '''Add spectral indices if related bands are available.'''
        band_idx = self.manifest['image_band_map']
        nodata = self.manifest['image_nodata']

        red = _Calc.mask(self.data.image[band_idx['red']], nodata)

        # add spectral indices on demand
        spectrals: list[numpy.ndarray] = []
        next_idx = self.data.image.shape[0]
        if 'ndvi' in indices:
            nir = _Calc.mask(self.data.image[band_idx['nir']], nodata)
            spectrals.append(_Calc.ndvi(nir, red, nodata))
            band_idx['ndvi'] = next_idx
            next_idx += 1

        if 'ndmi' in indices:
            nir = _Calc.mask(self.data.image[band_idx['nir']], nodata)
            swir1 = _Calc.mask(self.data.image[band_idx['swir1']], nodata)
            spectrals.append(_Calc.ndmi(nir, swir1, nodata))
            band_idx['ndmi'] = next_idx
            next_idx += 1

        if 'nbr' in indices:
            nir = _Calc.mask(self.data.image[band_idx['nir']], nodata)
            swir2 = _Calc.mask(self.data.image[band_idx['swir2']], nodata)
            spectrals.append(_Calc.nbr(nir, swir2, nodata))
            band_idx['nbr'] = next_idx
            next_idx += 1

        # add to image array
        if spectrals:
            added = numpy.stack(spectrals).astype(numpy.float32)
            self.data.image = numpy.append(self.data.image, added, axis=0)

    def _image_add_topography(self, features: list[str], pad: int) -> None:
        '''Add topographical metrics to the image array.'''
        _arrs = {
            'slope': numpy.zeros_like(self.data.image[0], dtype=numpy.float32),
            'cos_a': numpy.zeros_like(self.data.image[0], dtype=numpy.float32),
            'sin_a': numpy.zeros_like(self.data.image[0], dtype=numpy.float32),
            'tpi': numpy.zeros_like(self.data.image[0], dtype=numpy.float32),

        } # use first array as dim reference

        band_idx = self.manifest['image_band_map']
        nodata = self.manifest['image_nodata']

        # sanity check on image/padded image shape
        max_h, max_w = self.padded_dem.shape
        if not self.data.image[0].shape == (max_h - 2 * pad, max_w - 2 * pad):
            raise ValueError(
                f'Mismatch in image dimensions: {self.data.image[0].shape} vs'
                f'({max_h - 2 * pad}, {max_w - 2 * pad}), padding: {pad}'
            )

        # iterate through pixels from the original block in padded dem
        topos: list[numpy.ndarray] =  []
        next_idx = self.data.image.shape[0]
        for y in range(pad, max_h - pad):
            for x in range(pad, max_w - pad):

                if 'slope' in features:
                    # slope and aspect - pad neighbors, radius=1
                    pxs = _Calc.get_px_group(self.padded_dem, x, y, 1)
                    assert pxs.shape == (3, 3) # 3x3 window
                    (
                        _arrs['slope'][y - pad, x - pad],
                        _arrs['cos_a'][y - pad, x - pad],
                        _arrs['sin_a'][y - pad, x - pad]
                    ) = _Calc.slope_n_aspect(pxs, nodata)

                if 'tpi' in features:
                    # tpi - radius=pad-1 (default pad=8, radius=7)
                    pxs =  _Calc.get_px_group(self.padded_dem, x, y, pad - 1)
                    assert pxs.shape == (2* pad - 1, 2* pad - 1) # 15x15 window
                    _arrs['tpi'][y - pad, x - pad] = _Calc.tpi(pxs, nodata)

        # add to image array
        if 'slope' in features:
            topos.extend([_arrs['slope'], _arrs['cos_a'], _arrs['sin_a']])
            band_idx['slope'] = next_idx
            band_idx['cos_a'] = next_idx + 1
            band_idx['sin_a'] = next_idx + 2
            next_idx += 3

        if 'tpi' in features:
            topos.append(_arrs['tpi'])
            band_idx['tpi'] = next_idx
            next_idx +=1

        if topos:
            added = numpy.stack(topos, axis=0)
            self.data.image = numpy.append(self.data.image, added, axis=0)

    def _image_get_valid_mask(self):
        '''Get a valid mask for the whole block.'''
        # for image data: True where all image bands are valid
        nodata = self.manifest['image_nodata']
        invalid_img = self._get_image_invalid_mask(self.data.image, nodata)
        valid_img = ~numpy.any(invalid_img, axis=0)

        self.manifest['valid_ratios'].update({
            'image': float((numpy.sum(valid_img) / valid_img.size)),
        })
        self.data.valid_mask = valid_img

    def _image_get_stats(self):
        '''Per block stats for later aggregation using Welford's.'''
        # image_nodata
        nodata = self.manifest['image_nodata']
        # iterate through image channels
        for i, band in enumerate(self.data.image):
            # get where pixel is invalid and inverse to get valid pixels
            mask = self._get_image_invalid_mask(band, nodata)
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
            self.manifest['image_stats'][f'band_{i}'] = {
                'count': int(num), 'mean': float(mean), 'm2': float(mean_sq)
            }

    def _label_canonicalize(self) -> None:
        '''Normalize the label stack based on label specs.'''
        stack: list[numpy.ndarray] = []
        band_map = self.manifest['label_band_map']
        nodata = self.manifest['label_nodata']
        ignore_index = self.manifest['label_ignore_index']

        # iterate sorted bands sorted by index
        for name, i in sorted(band_map.items(), key=lambda item: item[1]):
            spec = self.lbl_specs[name]
            arr = self.data.label[i]

            # append base layer from original Class IDs with masking)
            self.manifest['label_ignore_cls'][name] = list(spec['ignore_cls'])
            to_ignore = list(spec['ignore_cls']) + [nodata, ignore_index]
            mask = ~numpy.isin(arr, to_ignore)
            shifted_arr = arr + (1 - spec['index_base'])
            normalized = numpy.where(mask, shifted_arr, ignore_index)

            # # skip if no reclass is defined for this label
            # reclass = spec.get('reclass')
            # if not reclass:
            #     continue
            # # grouping layers and children
            # group_layer = numpy.full_like(arr, ignore_index, dtype=arr.dtype)
            # for group_id, classes in reclass.items():
            #     mask = numpy.isin(arr, classes)
            #     group_layer[mask] = int(group_id) # modify in-place

            #     # create child slice: re-index original classes to 1..N
            #     child_arr = numpy.where(mask, arr, ignore_index)
            #     for k, cls_id in enumerate(classes, 1):
            #         child_arr[child_arr == cls_id] = int(k)
            #     stack.append(child_arr)

            # # append grouping layer last for this specification
            # stack.append(group_layer)

            # calculate valid pixel ratios
            valid = normalized != ignore_index
            ratio = float(valid.sum() / (valid.size)) if valid.size > 0 else 0.0
            self.manifest['valid_ratios'][name] = ratio

            # count unique values for the current head (classes 1..N)
            n_cls = spec['num_cls']
            self.manifest['label_num_cls'][name] = n_cls
            valids = normalized[valid].astype(numpy.int64)
            counts = numpy.bincount(valids, minlength=n_cls + 1)[1:n_cls + 1]
            self.manifest['label_count'][name] = [int(c) for c in counts]

            # entropy
            self.manifest['label_entropy'][name] = float(_Calc.entropy(counts))

            # attach class names and color map if provided
            if 'class_name' in spec and spec['class_name']:
                self.manifest['label_cls_names'][name] = list(spec['class_name'].values())
            if 'color_map' in spec and spec['color_map']:
                self.manifest['label_cls_clr_map'][name] = spec['color_map']

            stack.append(normalized)

        self.data.label = numpy.stack(stack, axis=0)

    # NOTE: to migrate to data prep
    # def _label_build_topology(self) -> None:
    #     '''
    #     Construct the label schema recorded in the block manifest.

    #     Populates self.manifest with hierarchy, class counts, and naming
    #     conventions derived from the label specifications.
    #     '''
    #     # containers
    #     num_cls: dict[str, int] = {}
    #     ignore_cls: dict[str, list[int]] = {}
    #     parent_map: dict[str, str | None] = {}
    #     parent_cls_map: dict[str, int | None] = {}
    #     label_names: dict[str, list[str]] = {}

    #     # iterate label specs
    #     for name, spec in self.lbl_specs.items():
    #         cls_name = spec.get('class_name', {})
    #         index_base = spec.get('index_base', 1)

    #         # base
    #         num_cls[name] = spec['num_cls']
    #         ignore_cls[name] = spec['ignore_cls']
    #         parent_map[name] = None
    #         parent_cls_map[name] = None
    #         label_names[name] = [
    #             cls_name.get(str(j + index_base), f'cls_{j + 1}')
    #             for j in range(spec['num_cls'])
    #         ]

    #         # skip if no reclass is defined for this label
    #         reclass = spec.get('reclass')
    #         if not reclass:
    #             continue

    #         # children
    #         grp_name = f'{name}_groups' # fallback genric name
    #         reclass_name = spec.get('reclass_name', {})
    #         for gid, classes in reclass.items():
    #             child_name = reclass_name.get(gid, f'{grp_name}_{gid}')
    #             num_cls[child_name] = len(classes)
    #             ignore_cls[child_name] = []
    #             parent_map[child_name] = grp_name
    #             parent_cls_map[child_name] = int(gid)
    #             label_names[child_name] = [
    #                 cls_name.get(str(c), f'cls_{c}')
    #                 for c in classes
    #             ]

    #         # parent
    #         num_cls[grp_name] = len(reclass)
    #         ignore_cls[grp_name] = []
    #         parent_map[grp_name] = None
    #         parent_cls_map[grp_name] = None
    #         label_names[grp_name] = [
    #             reclass_name.get(gid, f'grp_{gid}')
    #             for gid in sorted(reclass.keys(), key=int)
    #         ]

    #     # populate meta dict
    #     self.manifest['label_num_cls'] = num_cls
    #     self.manifest['label_ignore_cls'] = ignore_cls
    #     self.manifest['label_parent'] = parent_map
    #     self.manifest['label_parent_cls'] = parent_cls_map
    #     self.manifest['label_names'] = label_names

    #     label_taxonomy: dict[str, typing.Any] = {}
    #     for name, spec in self.lbl_specs.items():
    #         if 'taxonomy' in spec and spec['taxonomy']:
    #             label_taxonomy[name] = spec['taxonomy']
    #     if label_taxonomy:
    #         self.manifest['label_taxonomy'] = label_taxonomy

    @staticmethod
    def _get_image_invalid_mask(
        image: numpy.ndarray,
        nodata: float | None
    ):
        '''Return True where image values are invalid.'''
        invalid = numpy.isnan(image)

        if nodata is not None:
            if not (isinstance(nodata, float) and numpy.isnan(nodata)):
                invalid |= numpy.isclose(image, nodata)

        return invalid

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
    def get_px_group(arr, x, y, rr):
        '''Get neighbouring pixels as an array'''
        return arr[slice(y - rr, y + rr + 1), slice(x - rr, x + rr + 1)]

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
        # deterministically handle cos/sin when slope == 0
        if slope == 0.0:
            return 0.0, 1.0, 0.0
        # calculate aspect angle in radians
        aspect_rad = numpy.arctan2(dz_dy, -dz_dx)
        if aspect_rad < 0:
            aspect_rad += 2 * numpy.pi  # normalize to [0, 2π]
        # compute cosine and sine of aspect
        cos_aspect = numpy.cos(aspect_rad)
        sin_aspect = numpy.sin(aspect_rad)
        return slope, cos_aspect, sin_aspect

    @staticmethod
    def tpi(arr, nodata):
        '''Returns Topographical Position Index (TPI) from a DEM.'''
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
