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

This module defines a lightweight container/builder for a single raster
*block* (window). It accepts arrays already read elsewhere and augments
them with derived features and block-level metadata—without performing
any raster I/O.

**Key responsibilities**
- Ingest per-window arrays: image (C,H,W), optional label (1,H,W → H,W),
  and a DEM padded on all sides (for neighborhood topo metrics).
- Add derived channels: NDVI/NDMI/NBR (from band assignments) and slope,
  cos(aspect), sin(aspect), TPI (from the padded DEM).
- Build hierarchical labels (layer-1 and layer-2 reclasses), compute
  class counts and Shannon entropy.
- Compute a per-block valid mask, and per-band stats (count, mean, M2)
  for global aggregation (e.g., Welford's online algorithm).
- Save/load a block to/from a single .npz artifact.

**Assumptions**
- Heavy lifting (reading rasters, windowing, padding) is done upstream;
  this module only consumes NumPy arrays and metadata.
- Shapes: image is (C,H,W); label, if provided, is (1,H,W) and squeezed
  to (H,W); DEM is larger than (H,W) by 2*dem_pad in each dimension.
- Required metadata includes (non-exhaustive):
  * image_nodata, label_nodata, ignore_label, dem_pad
  * band_assignment with at least: 'red', 'nir', 'swir1', 'swir2'
  * label_num_classes, label_to_ignore, label_reclass_map
  * block_name
- Nodata handling relies on np.isnan / np.isclose and masked arrays;
  ensure nodata values are correctly specified.
- Topographic metrics use small neighborhoods and computed per pixel.
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
class BlockMeta(typing.TypedDict):
    '''Defines the shape of a block meta dictionary.'''
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
    label_masked: numpy.ndarray = dataclasses.field(init=False)
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

    A `DataBlock` represents a single raster window compiled from arrays
    read upstream. It augments raw inputs with derived spectral indices,
    topographic metrics, label structures, and block-level statistics.

    This class performs no raster I/O during creation; all inputs are
    provided as NumPy arrays. Blocks can be serialized to and restored
    from a single `.npz` artifact.

    Typical usage:
      - `create()` to build a new block from arrays
      - `save()` / `load()` for persistence
      - `normalize_image()` for post hoc iamge normalization
    '''

    def __init__(self, **kwargs):
        '''
        Initialize an empty data block with placeholder data and meta.

        A `DataBlock` can be instantiated without arguments and later
        populated via `create()` or `load()`.
        '''

        # init with empty block data
        self.data = _BlockArrays()
        # meta dict with default foo values
        self.meta: BlockMeta = {
            'block_name': '',
            'has_label': True,
            'ignore_index': kwargs.get('ignore_index', 255),
            'valid_ratios': {},
            'label_nodata': 0,
            'label_num_cls': 0,
            'label_ignore_cls': [],
            'label_reclass_map': {},
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
        meta: BlockMeta,
    ) -> 'DataBlock':
        '''
        Create a data block from in-memory raster arrays.

        Args:
            img_arr: Image array of shape (C, H, W).
            lbl_arr: Optional label array of shape (1, H, W). If None,
                the block is treated as unlabeled.
            padded_dem: DEM array padded on all sides for neighborhood
                calculations.
            meta: Block-level metadata and configuration.

        Returns:
            DataBlock: The populated block instance (returned for
                chaining).
        ----------------------------------------------------------------
        Notes: This method derives spectral indices, topographic metrics,
        label hierarchies, valid masks, and per-block statistics.
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
            self.data.label_masked = numpy.array([1])
            self.meta['has_label'] = False

        # process data sequence
        # image bands related
        self._add_spectral_indices()
        self._add_topographical_metrics()
        # labels related
        self._build_label_hierarchy()
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
        Load data from existing .npz file to populate a class instance.

        Args:
            fpath: Path to a serialized block artifact.

        Returns:
            DataBlock: The populated block instance.
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
        Save the block data to a compressed `.npz` file.

        Args: Output file path. Existing files will be overwritten.
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

    def _build_label_hierarchy(self) -> None:
        '''Prep the hierarchy of labels according to metadata.'''

        # skip if label is not provided
        if not self.meta['has_label']:
            return

        # get value from meta
        ignore_index = self.meta['ignore_index']
        label_nodata = self.meta['label_nodata']
        label_to_ignore = self.meta['label_ignore_cls']
        reclass_map = self.meta['label_reclass_map']

        # fill invalid pixels with ignore label
        label_to_ignore = list(label_to_ignore) # avoid modifying the meta
        label_to_ignore.append(label_nodata)
        label_to_ignore = [x for x in label_to_ignore if x is not None]

        # collection of layers
        layer1_mask = ~numpy.isin(self.data.label, label_to_ignore)
        layer1_valid = numpy.where(layer1_mask, self.data.label, ignore_index)
        fn_stack = [layer1_valid] # layer1 as the first element of the list

        # if no reclass, only keep layer1 == original label
        if not reclass_map:
            self.data.label_masked = numpy.stack(fn_stack, axis=0)
            self.meta['valid_ratios'].update({
                'layer1': float(numpy.sum(layer1_mask) / layer1_valid.size)
            })
            return

        # iterate through layer classes in reclass map
        for band_num, classes in reclass_map.items():
            # mask to the current layer 1 class (as the layer2 subclasses)
            _mask = numpy.isin(self.data.label, classes)
            # in-place reclass relevant pixels in layer1
            fn_stack[0][_mask] = int(band_num)
            # create a layer 2 array for current layer1 class
            layer2_new = numpy.where(_mask, self.data.label, ignore_index)
            # reclass from 1 to n
            for i, cls in enumerate(classes, 1):
                layer2_new[layer2_new == cls] = int(i)
            # append the new layer 2 array to the stack
            fn_stack.append(layer2_new)

        # stack all the arrays and assign to attr
        self.data.label_masked = numpy.stack(fn_stack, axis=0)

        # add metadata
        self.meta['valid_ratios'].update({
            'layer1': float(numpy.sum(layer1_mask) / layer1_valid.size)
        })
        for i in range(len(reclass_map)):
            _valid = fn_stack[i + 1] != ignore_index
            self.meta['valid_ratios'].update({
                f'layer2_{i + 1}': numpy.sum(_valid) / layer1_valid.size
            })

    def _count_label_classes(self) -> None:
        '''Count present label values and calculate entropy.'''

        # skip if label is not provided
        if not self.meta['has_label']:
            return

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
        lyrs = numpy.concatenate(
            [self.data.label[None, :, :], self.data.label_masked], axis=0
        ) # e.g., (7, 256, 256)

        # sanity check
        assert len(n_classes) == len(lyrs)

        # iterate label layers
        for i, band in enumerate(lyrs):
            # count unique values for each label layer
            label_unique = numpy.arange(1, n_classes[i] + 1) # start from 1
            filtered = band[numpy.isin(band, label_unique)]
            uniques, counts = numpy.unique(filtered, return_counts=True)

            # convert to list. avoid using arr.tolist()
            uniques = [int(_) for _ in uniques]
            counts = [int(_) for _ in counts]

            # shannon entropy
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
                self.meta['label_count'].update({'original_label': cc})
                self.meta['label_entropy'].update({'original_label': ent})
            elif i == 1:
                self.meta['label_count'].update({'layer1': cc})
                self.meta['label_entropy'].update({'layer1': ent})
            else:
                self.meta['label_count'].update({f'layer2_{i - 1}': cc})
                self.meta['label_entropy'].update({f'layer2_{i - 1}': ent})

    def _get_block_valid_mask(self):
        '''Get a valid mask for the whole block.'''

        # get image nodata and ignore label index
        ignore_index = self.meta['ignore_index']
        img_nodata = self.meta['image_nodata']

        # for label data
        # if provided, locs where label layer1 is valid (not ignore)
        if self.meta['has_label']:
            valid_label = self.data.label_masked[0] != ignore_index
        # otherwise no effect
        else:
            valid_label = True # easy broadcast

        # for image data
        # locs where image all bands are valid and not nodata
        invalid_img = (
            numpy.isnan(self.data.image) |
            numpy.isclose(self.data.image, img_nodata)
        )
        valid_img = ~numpy.any(invalid_img, axis=0) # shape (256, 256)

        # final mask (combined)
        self.data.valid_mask = valid_label & valid_img # (256, 256)

        # add ratios to meta
        self.meta['valid_ratios'].update({
            'image': float((numpy.sum(valid_img) / valid_img.size)),
            'block': float((numpy.sum(self.data.valid_mask) / valid_img.size))
        })

    def _get_block_image_stats(self):
        '''Per block stats for later aggregation using Welford's.'''

        # parse
        image_nodata = self.meta['image_nodata']

        #
        for i, band in enumerate(self.data.image):
            # get where pixel is not valid
            if isinstance(image_nodata, float) and numpy.isnan(image_nodata):
                mask = numpy.isnan(band)
            else:
                mask = numpy.isclose(band, image_nodata)
            # inverse to get valid pixels
            valid = band[~mask]
            # 3 block stats to be done: nb, mb, m2b
            num = valid.size
            # if no valid pixels
            if num == 0:
                # safe neutral values for Welford aggregation
                mean = 0.0
                mean_sq = 0.0
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
