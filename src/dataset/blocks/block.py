'''
DataBlock: compile per-block arrays and metadata for geospatial ML.

This module defines a lightweight container/builder for a single raster
*block* (window). It accepts arrays already read elsewhere and augments
them with derived features and block-level metadata—without performing
any raster I/O.

Key responsibilities
- Ingest per-window arrays: image (C,H,W), optional label (1,H,W → H,W),
  and a DEM padded on all sides (for neighborhood topo metrics).
- Add derived channels: NDVI/NDMI/NBR (from band assignments) and slope,
  cos(aspect), sin(aspect), TPI (from the padded DEM).
- Build hierarchical labels (layer-1 and layer-2 reclasses), compute
  class counts and Shannon entropy.
- Compute a per-block valid mask, and per-band stats (count, mean, M2)
  for global aggregation (e.g., Welford's online algorithm).
- Save/load a block to/from a single .npz artifact.

Assumptions
- Heavy lifting (reading rasters, windowing, padding) is done upstream;
  this module only consumes NumPy arrays and metadata.
- Shapes: image is (C,H,W); label, if provided, is (1,H,W) and squeezed
  to (H,W); DEM is larger than (H,W) by 2*dem_pad in each dimension.
- Required metadata includes (non-exhaustive):
  * image_nodata, label_nodata, ignore_label, dem_pad
  * band_assignment with at least: 'red', 'nir', 'swir1', 'swir2'
  * label1_num_classes, label1_to_ignore, label1_reclass_map,
    label1_class_name, label1_reclass_name
  * block_name, block_shape
- Nodata handling relies on np.isnan / np.isclose and masked arrays;
  ensure nodata values are correctly specified.
- Topographic metrics use small neighborhoods and are computed per pixel;
  for very large blocks, consider vectorization/acceleration if needed.

Public API
- DataBlock.create_from_rasters(img_arr, lbl_arr, dem_padded, meta) -> DataBlock
- DataBlock.save_npz(path, compress=True) / DataBlock.load_from_npz(path)
- DataBlock.normalize_image(global_stats) -> (min, max)
'''

from __future__ import annotations
# standard imports
import dataclasses
import json
import math
import typing
# third party imports
import numpy

class DataBlock:
    '''Data block compiled from input label and image rasters.'''

    def __init__(self):
        '''Can be instantiate without arguments.'''

        # init with empty block data and meta
        self.data = _Data()
        self.meta: dict[str, typing.Any]

    # -----------------------------public methods-----------------------------
    def create_from_rasters(
            self,
            img_arr: numpy.ndarray,
            lbl_arr: numpy.ndarray | None,
            dem_padded: numpy.ndarray,
            meta: dict[str, typing.Any]
        ) -> 'DataBlock':
        '''
        Create a new block instance from raw input raster(s).
        '''

        # assign meta
        self.meta = meta
        # assign image
        self.data.image = img_arr.astype(numpy.float32) # remote sensing default
        self.data.image_dem_padded = dem_padded.astype(numpy.float32) # padded
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
            self.data.label = numpy.empty([1])
            self.data.label_masked = numpy.empty([1])
            self.meta['has_label'] = False

        # process data sequence
        # image bands related
        self._image_add_spec_indices()
        self._image_add_topo_metrics()
        # labels related
        self._labels_get_structure()
        self._label_count_classes()
        # block-wise
        self._get_block_valid_mask()
        self._get_block_image_stats()
        # metadata
        self.__reorder_meta()
        # sanity check self.data
        self.data.validate(skip_attr='image_normalized') # to be populated later

        # return self to allow chained calls
        return self

    def load_from_npz(
            self,
            fpath: str
        ) -> 'DataBlock':
        '''
        Load data from existing .npz file to populate a class instance.
        '''

        # load and set attributes
        loaded = numpy.load(fpath, allow_pickle=True)
        for key in loaded:
            value = loaded[key]
            # 'meta' is a pickled dict, convert from 0-d array
            if key == 'meta' and isinstance(value, numpy.ndarray):
                value = value.item()
            # set value
            try:
                setattr(self.data, key, value)
            except AttributeError:
                continue # don't anything here yet

        # return self to allow chained calls
        return self

    def save_npz(
            self,
            fpath: str,
            compress: bool=True
        ) -> None:
        '''Save block data as an `.npz` file. Will overwrite.'''

        # sanity check
        assert fpath.endswith('.npz')
        # directly write self.data to npz file
        save_data = vars(self.data)
        # save file - allow pickle to write meta dict
        if compress:
            numpy.savez_compressed(fpath, allow_pickle=True, **save_data)
        else:
            numpy.savez(fpath, allow_pickle=True, **save_data)

    def recalc_stats(
            self,
            fpath: str
        ) -> None:
        '''
        Re-calculate block-level stats and save in-place.

        Rarely used when some blocks contain corrupted stats.
        '''

        self._get_block_image_stats()
        self.save_npz(fpath)

    def normalize_image(
            self,
            global_stats: dict[str, dict[str, int | float]],
        ) -> tuple[float, float]:
        '''
        Normalize block-level image bands using provided global stats.

        Such global stats are typically aggregated from a set of blocks
        and this method is called to added normalized image channels to
        a block post hoc.
        '''

        # assertion
        assert len(global_stats) == self.data.image.shape[0]

        # init data attribute, inherit dtype float32
        self.data.image_normalized = numpy.empty_like(self.data.image)

        # normalize each band
        for i, (band, stats) in enumerate(global_stats.items()):
            # sanity check - dict keys from band_0
            assert band.lstrip('band_') == str(i)
            # get global stats from input
            g_mean = stats['current_mean']
            g_std = stats['std'] if stats['std'] != 0 else 1
            # get image band and replace invalid pixels with global mean
            img_band = self.data.image[i]
            img_band = numpy.where(self.data.valid_mask, img_band, g_mean)
            # normalize band
            self.data.image_normalized[i] = (img_band - g_mean) / g_std

        # sanity check the minmax of normalized image
        mmin = self.data.image_normalized.min().item()
        mmax = self.data.image_normalized.max().item()
        # return
        return mmin, mmax

    # ----------------------------internal methods----------------------------
    def _image_add_spec_indices(self) -> None:
        '''Add spectral indices using loaded Landsat bands.'''

        # skip if already created
        if self.meta.get('spectral_indices_added', False):
            return

        # retrieve from meta
        spec_bands = self.meta.get('band_assignment', {})
        nodata = self.meta.get('image_nodata', numpy.nan)

        # assertions
        assert all(k in spec_bands for k in ['red', 'nir', 'swir1', 'swir2'])

        # mask off nodata pixels to avoid overflow
        # 32 bit increased to 64 bit float
        red = _Calc.mask(self.data.image[spec_bands['red']], nodata)
        nir = _Calc.mask(self.data.image[spec_bands['nir']], nodata)
        swir1 = _Calc.mask(self.data.image[spec_bands['swir1']], nodata)
        swir2 = _Calc.mask(self.data.image[spec_bands['swir2']], nodata)

        # add data to class
        self.meta['spectral_indices_added'] = ['NDVI', 'NDMI', 'NBR']
        n = len(self.meta['band_assignment'])
        self.meta['band_assignment'].update({
            'NDVI': n,
            'NDMI': n + 1,
            'NBR': n + 2
        })
        add_indices = numpy.stack([
            _Calc.ndvi(nir, red, nodata),
            _Calc.ndmi(nir, swir1, nodata),
            _Calc.nbr(nir, swir2, nodata)
        ]).astype(numpy.float32)
        self.data.image = numpy.append(self.data.image, add_indices, axis=0)

    def _image_add_topo_metrics(self) -> None:
        '''Add topographical metrics to the image array.'''

        # skip if already added
        if self.meta.get('topo_metrics_added', False):
            return

        # get vars
        pad = self.meta['dem_pad']
        nodata = self.meta.get('image_nodata', numpy.nan)
        max_h, max_w = self.data.image_dem_padded.shape
        # sanity check
        assert self.data.image[0].shape == (max_h - 2 * pad, max_w - 2 * pad)

        # prep metrics to add
        slope = numpy.zeros_like(self.data.image[0], dtype=numpy.float32)
        cos_a = numpy.zeros_like(self.data.image[0], dtype=numpy.float32)
        sin_a = numpy.zeros_like(self.data.image[0], dtype=numpy.float32)
        tpi = numpy.zeros_like(self.data.image[0], dtype=numpy.float32)

        # iterate through pixels from the original block in padded dem
        for y in range(pad, max_h - pad):
            for x in range(pad, max_w - pad):
                # slope and aspect - 8 neighbors, radius 1
                pxs = _Calc.get_px_group(self.data.image_dem_padded, x, y, 1)
                slope[y - pad, x - pad], cos_a[y - pad, x - pad], \
                    sin_a[y - pad, x - pad] = _Calc.slope_n_aspect(pxs, nodata)
                # tpi - 224 neighbors, radius 7
                pxs =  _Calc.get_px_group(self.data.image_dem_padded, x, y, 7)
                tpi[y  - pad, x - pad] = _Calc.tpi(pxs, nodata)

        # add to metadata
        self.meta['topo_metrics_added'] = [
            'slope', 'cos_aspect', 'sin_aspect', 'tpi'
        ]
        n = len(self.meta['band_assignment'])
        self.meta['band_assignment'].update({
            'slope': n,
            'cos_aspect': n + 1,
            'sin_aspect': n + 2,
            'tpi': n + 3
        })

        # add to self.image
        to_add = numpy.stack([slope, cos_a, sin_a, tpi], axis=0)
        self.data.image = numpy.append(self.data.image, to_add, axis=0)

    def _labels_get_structure(self) -> None:
        '''Prep the hierarchy of labels according to metadata.'''

        # skip if label is not provided
        if not self.meta['has_label']:
            return

        # get kwargs
        label_nodata = self.meta.get('label_nodata', 0) # ignore 0 -> safe
        label1_to_ignore = self.meta.get('label1_to_ignore', [])
        ignore_label = self.meta.get('ignore_label', 255)
        label1_reclass = self.meta.get('label1_reclass_map', {})

        # fill invalid pixels with ignore label
        label1_to_ignore = list(label1_to_ignore) # avoid modifying the meta
        label1_to_ignore.append(label_nodata)
        label1_to_ignore = [x for x in label1_to_ignore if x is not None]
        layer1_mask = ~numpy.isin(self.data.label, label1_to_ignore)
        layer1_valid = numpy.where(layer1_mask, self.data.label, ignore_label)

        # collection of final layers
        fn_stack = [layer1_valid] # layer1 as the first element of the list

        # iterate through layer1 classes in reclass map
        for band_num, classes in label1_reclass.items():
            # mask to the current layer 1 class (as the layer2 subclasses)
            _mask = numpy.isin(self.data.label, classes)
            # in-place reclass relevant pixels in layer1
            fn_stack[0][_mask] = int(band_num)
            # create a layer 2 array for current layer1 class
            layer2_new = numpy.where(_mask, self.data.label, ignore_label)
            # reclass from 1 to n
            for i, cls in enumerate(classes, 1):
                layer2_new[layer2_new == cls] = int(i)
            # append the new layer 2 array to the stack
            fn_stack.append(layer2_new)

        # stack all the arrays and assign to attr
        self.data.label_masked = numpy.stack(fn_stack, axis=0)

        # add metadata
        self.meta['valid_pixel_ratio'] = {
            'layer1': numpy.sum(layer1_mask) / layer1_valid.size
        }
        for i in range(len(label1_reclass)):
            _valid = fn_stack[i + 1] != ignore_label
            self.meta['valid_pixel_ratio'].update({
                f'layer2_{i + 1}': numpy.sum(_valid) / layer1_valid.size
            })

    def _label_count_classes(self) -> None:
        '''Count present label values and calculate entropy.'''

        # skip if label is not provided
        if not self.meta['has_label']:
            return

        # write to following entries for meta
        self.meta['label_count'] = {}
        self.meta['label_entropy'] = {}

        # supposed number of classes for each label layer
        n_classes = [
            self.meta['label1_num_classes'], # count of original label
            len(self.meta['label1_reclass_map']) # count of reclassed lyr1
        ]
        n_classes.extend([
            len(v) for v in self.meta['label1_reclass_map'].values()
        ]) # counts of lyr2 groups

        # all arrays to be counted
        lyrs = numpy.concatenate(
            [self.data.label[None, :, :], self.data.label_masked], axis=0
        ) # e.g., (7, 256, 256)

        # sanity check
        assert len(n_classes) == len(lyrs)

        # iteration
        for i, band in enumerate(lyrs):
            # count unique values for each label layer
            label_unique = numpy.arange(1, n_classes[i] + 1) # start from 1
            filtered = band[numpy.isin(band, label_unique)]
            uniques, counts = numpy.unique(filtered, return_counts=True)

            # convert to list. avoid using arr.tolist()
            uu = [int(_) for _ in uniques]
            cc = [int(_) for _ in counts]

            # shannon entropy
            ent = 0.0
            for _ in cc:
                p = _ / sum(cc)
                ent -= p * math.log2(p)

            # assign zero count to no_show classes
            counts = []
            for _ in range(n_classes[i]):
                idx = _ + 1
                if idx in uu:
                    count_idx = uu.index(idx)
                    counts.append(cc[count_idx])
                else:
                    counts.append(0)

            # add to metadata
            if i == 0:
                self.meta['label_count']['original_label'] = counts
                self.meta['label_entropy']['original_label'] = ent
            elif i == 1:
                self.meta['label_count']['layer1'] = counts
                self.meta['label_entropy']['layer1'] = ent
            else:
                self.meta['label_count'][f'layer2_{i - 1}'] = counts
                self.meta['label_entropy'][f'layer2_{i - 1}'] = ent

    def _get_block_valid_mask(self):
        '''Get a valid mask for the whole block.'''

        # get image nodata and ignore label index
        img_nodata = self.meta.get('image_nodata', numpy.nan)
        ignore_label = self.meta.get('ignore_label', 255)

        # for label data
        # if provided, locs where label layer1 is valid (not ignore)
        if self.meta['has_label']:
            valid_label = self.data.label_masked[0] != ignore_label
        # otherwise no effect
        else:
            valid_label = True # easy broadcast

        # for image data
        # locs where image all bands are valid and not nodata
        invalid_img = numpy.isnan(self.data.image) | \
            numpy.isclose(self.data.image, img_nodata)
        valid_img = ~numpy.any(invalid_img, axis=0) # shape (256, 256)

        # final mask (combined)
        self.data.valid_mask = valid_label & valid_img # (256, 256)

        # add ratios to meta
        self.meta['valid_pixel_ratio'].update({
            'image': float((numpy.sum(valid_img) / valid_img.size)),
            'block': float((numpy.sum(self.data.valid_mask) / valid_img.size))
        })

    def _get_block_image_stats(self):
        '''Per block stats for later aggregation using Welford's.'''

        # parse
        image_nodata = self.meta.get('image_nodata', numpy.nan)
        # create dict for block stats
        self.meta['block_image_stats'] = {}
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
            mean = numpy.mean(valid)
            mean_sq = numpy.sum((valid - mean) ** 2)
            # give to self.image_stats
            self.meta['block_image_stats'][f'band_{i}'] = {
                'count': int(num), 'mean': float(mean), 'm2': float(mean_sq)
            }

    def __reorder_meta(self):
        '''Reorder meta dict keys.'''

        # block
        _block = [
            'block_name',
            'block_shape',
            'valid_pixel_ratio',
            'has_label'
        ]
        # label
        _label = [
            'label_nodata',
            'ignore_label',
            'label1_num_classes',
            'label1_to_ignore',
            'label1_class_name',
            'label1_reclass_map',
            'label1_reclass_name',
            'label_count',
            'label_entropy'
        ]
        # image
        _image = [
            'image_nodata',
            'dem_pad',
            'band_assignment',
            'spectral_indices_added',
            'topo_metrics_added',
            'block_image_stats'
        ]

        image_label = _block + _label + _image
        image_only = _block + _image

        # reorder with sanity checks
        if self.meta['has_label']:
            assert set(image_label) == set(self.meta.keys()), \
                f'{image_label}\n {self.meta.keys()}'
            self.meta = {k: self.meta[k] for k in image_label}
        else:
            assert set(image_only) == set(self.meta.keys()), \
                f'{image_only}\n {self.meta.keys()}'
            self.meta = {k: self.meta[k] for k in image_only}
# ---------------------------------Caculators---------------------------------
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

# ------------------------------block data class------------------------------
@dataclasses.dataclass
class _Data:
    '''Simple dataclass for block-wise image/label data.'''

    label: numpy.ndarray = dataclasses.field(init=False)
    label_masked: numpy.ndarray = dataclasses.field(init=False)
    image: numpy.ndarray = dataclasses.field(init=False)
    image_dem_padded: numpy.ndarray = dataclasses.field(init=False)
    image_normalized: numpy.ndarray = dataclasses.field(init=False)
    valid_mask: numpy.ndarray = dataclasses.field(init=False)

    def __repr__(self) -> str:
        lines = ['Block summary\n']
        lines.append('-' * 70)
        # iterate attributes from the dataclass
        for field in dataclasses.fields(self):
            value = getattr(self, field.name)
            if isinstance(value, numpy.ndarray):
                lines.append(f'{field.name}: ')
                lines.append(f'Array shape: {value.shape}')
                lines.append(f'Array dtype: {value.dtype}')
                lines.append('-' * 70)
            elif isinstance(value, dict):
                lines.append(f'{field.name}: ')
                lines.append(f'{json.dumps(value, indent=4)}')
                lines.append('-' * 70)
            else:
                lines.append(f'{field.name}: ')
                lines.append(f'{value}')
                lines.append('-' * 70)
        # return lines
        return '\n'.join(lines)

    def validate(self, skip_attr: str | None=None):
        '''Validate if all attr has been populated.'''
        for field in dataclasses.fields(self):
            if skip_attr is not None and field.name == skip_attr:
                setattr(self, skip_attr, numpy.empty([1])) # place holder
            if not hasattr(self, field.name):
                raise ValueError(f"{field.name} has not been populated yet")
