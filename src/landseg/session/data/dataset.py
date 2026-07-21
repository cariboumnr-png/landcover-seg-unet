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
Dataset utilities for block-tiling segmentation with optional domains.

This module turns a collection of square block `.npz` files into fixed-
size patch samples suitable for training/evaluating segmentation models.

Key concepts:
- Block tiling: Each block of size `block_size x block_size` is split
    into uniform patches of size `patch_size x patch_size` in a
    deterministic scanline order.
- Two modes:
    * preload=True: eagerly load & patchify all blocks into RAM
        (i.e., for small static sets).
    * preload=False: lazily map a global index to (block_idx, patch_idx)
        and stream blocks via LRU-like cache (i.e., for large sets).
- Domain features: An optional per-block `domain_map` can attach
    categorical IDs (int) or continuous vectors (list[float]) to each
    returned sample.

Classes:
    BlockConfig: Configuration for tiling, simple flips, and optional
        domain map.
    MultiBlockDataset: Dataset over multiple `.npz` blocks producing
        (x, y, dom).
    _BlockDataset: Internal helper to patchify a single block and
        prepare its (x, y, dom).
    _CacheDict: Small fixed-size, LRU-like cache used in streaming mode.

Returns (per sample):
    tuple(
        x: torch.FloatTensor[C, ps, ps],\n
        y: torch.LongTensor[ps, ps],\n
        dom: dict[str, torch.Tensor]
    )

Notes:
- Domain tensors are attached per block; image/label flips do not\
    affect domain.
- For uniform patches per block, global index mapping uses integer\
    division; a cumulative index array is unnecessary.
'''


from __future__ import annotations
# standard imports
import collections
import dataclasses
# third-party imports
import numpy
import torch
import torch.utils.data
import torchvision.transforms.functional
# local imports
import landseg.session.common.alias as alias


@dataclasses.dataclass
class BlockConfig:
    '''
    Configuration for partitioning data blocks into patches and attaching
    optional per-block domain features.

    The configuration defines how square image blocks are partitioned
    into uniformly sized patches, identifies the image and label arrays
    stored in each block `.npz` file, and optionally supplies per-block
    domain metadata to be attached to every returned sample.

    Attributes:
        block_size: Side length (pixels) of each square data block.
        patch_size: Side length (pixels) of each square patch.
        image_key: Name of the image array in each `.npz` file.
        label_key: Name of the label array in each `.npz` file.
        ids_domain: Mapping {blk name: categorical domain ID}.
        vec_domain: Mapping {blk name: continuous domain feature vector}.

    Raises:
        ValueError: If `block_size` is smaller than `patch_size`, or if
            `block_size` is not evenly divisible by `patch_size`.
    '''
    block_size: int
    patch_size: int
    image_key: str
    label_key: str
    ids_domain: dict[str, int] | None = None
    vec_domain: dict[str, list[float]] | None = None

    def __post_init__(self):
        if self.block_size < self.patch_size:
            raise ValueError('Data block size must > dataset patch size')

        if self.block_size % self.patch_size != 0:
            raise ValueError('Data block size must be divisible by patch size')

    @property
    def patch_per_dim(self) -> int:
        '''Return the number of patches along each spatial dimension.'''
        return int(self.block_size // self.patch_size)

    @property
    def patch_per_blk(self) -> int:
        '''Return total number of patches extracted from one block.'''
        return self.patch_per_dim ** 2


@dataclasses.dataclass
class _MultiBlockData:
    '''Small container for multiblock data.'''
    img: numpy.ndarray | _CacheDict = dataclasses.field(init=False)
    lbl: numpy.ndarray | _CacheDict = dataclasses.field(init=False)
    dom: list[alias.TorchDict] | _CacheDict = dataclasses.field(init=False)


class MultiBlockDataset(torch.utils.data.Dataset):
    '''
    Unified dataset over multiple block .npz files with patch extraction
    and optional domain features.

    This dataset abstracts a set of block files and yields fixed-size
    patches `(x, y, dom)` for training segmentation models.

    Operating modes:
    * preload=True: eagerly load, patchify, and concatenate all blocks
        into RAM. Samples are indexed directly from contiguous arrays.
    * preload=False: lazily map a global index to (block_idx, patch_idx),
        loading patchified blocks on demand into a small LRU-like cache.

    The optional `domain_map` in `BlockConfig` is consulted per block;
    when present, a per-block domain dict is attached to each returned
    patch. Integer values are converted to `torch.long` and list values
    to `torch.float32`.

    Attributes:
        fpaths (list[str]):
            List of block file paths (`.npz`).
        preload (bool):
            Whether to preload or stream.
        block_config (BlockConfig):
            Block config see the dataclass.
        data (_MultiBlockData):
            Storage for images, labels, and domains. In preload mode,
            image and label arrays are concatenated into NumPy arrays
            and domains are stored as a list. In streaming mode, each
            field is backed by a `_CacheDict`.

    Raises:
        ValueError: If patch extraction fails for a block due to shape
            or data issues.
    '''

    def __init__(
        self,
        block_src: dict[str, str],
        config: BlockConfig,
        *,
        preload: bool = False,
        augment_flip: bool = False,
        **kwargs
    ):
        '''
        Initialize the dataset over multiple data blocks (.npz files).

        Args:
            block_src:
                Mapping of block name to `.npz` path.
            config:
                Dataset tiling and domain configuration.
            preload:
                If True, eagerly load all blocks.
            augment_flip:
                Apply random horizontal/vertical flips when labels are
                present.
            blk_cache_num:
                Maximum number of streamed blocks to retain in cache.
                Passed via `kwargs`.
        '''
        super().__init__()

        self.blks = block_src
        self.cfg = config
        self.preload = preload
        self.aug_flip = augment_flip

        self.data = _MultiBlockData()
        self._counter: tuple[int, int] = (0, 0)

        if self.preload: # load all files into one dataset
            _imgs: list[numpy.ndarray] = []
            _lbls: list[numpy.ndarray] = []
            self.data.dom = []
            # no progress bar if logger is silent
            for blk_name, blk_fpath in block_src.items():
                blk_data = _BlockDataset(
                    blk_fpath,
                    config,
                    self._get_domain(blk_name),
                    augment_flip=self.aug_flip
                )
                _imgs.append(blk_data.imgs)
                _lbls.append(blk_data.lbls)
                self.data.dom.extend([blk_data.domain] * config.patch_per_blk)
            # concatenate
            self.data.img = numpy.concatenate(_imgs, axis=0)
            self.data.lbl = numpy.concatenate(_lbls, axis=0)
            self._counter = len(block_src), 0

        else: # otherwise streaming
            blk_cache_num = kwargs.get('blk_cache_num', 16)
            # below not needed for uniform n
            # n =  self.block_config.patch_per_block
            # self.cumulative_i = [n * i for i in range(len(fpaths) + 1)]
            self.data.img = _CacheDict(maxsize=blk_cache_num)
            self.data.lbl = _CacheDict(maxsize=blk_cache_num)
            self.data.dom = _CacheDict(maxsize=blk_cache_num)
            self._counter = 0, len(block_src)

    def __len__(self):
        return len(self.blks) * self.cfg.patch_per_blk

    def __getitem__(self, idx: int) -> alias.DatasetItem:
        if self.preload:
            x = self.data.img[idx].astype(numpy.float32)  # [C, ps, ps]
            if (
                isinstance(self.data.lbl, numpy.ndarray) and
                self.data.lbl.ndim == 1
            ):
                y = numpy.array([1])
            else:
                y = self.data.lbl[idx].astype(numpy.int64)  # [ps, ps]
            dom = self.data.dom[idx] # dict[str, tensor]
            dom = {k: v.clone() for k, v in dom.items()} # new - no shared refs
        else:
            blk_idx, pch_idx = self._global_to_local(idx)
            self._load_block_to_cache(blk_idx)
            x = self.data.img[blk_idx][pch_idx]
            if (
                isinstance(self.data.lbl[blk_idx], numpy.ndarray) and
                self.data.lbl[blk_idx].ndim == 1
            ):
                y = numpy.array([1])
            else:
                y = self.data.lbl[blk_idx][pch_idx].astype(numpy.int64)  # [ps, ps]
            dom = self.data.dom[blk_idx] # domain at is batch-level
            dom = {k: v.clone() for k, v in dom.items()} # new - no shared refs
        # in case y is a placeholder
        if y.ndim != 1:
            y = torch.from_numpy(y).long()
        else:
            y = torch.empty(0, dtype=torch.long)
        return torch.from_numpy(x), y, dom

    @property
    def n_preloaded(self) -> int:
        '''Return number of blocks pre-loaded into RAM.'''
        return self._counter[0]

    @property
    def n_cached(self) -> int:
        '''Return number of blocks cached in RAM.'''
        return self._counter[1]

    def _global_to_local(self, idx: int) -> tuple[int, int]:
        '''Map global patch to block/patch indices (uniform blocks).'''
        blk_idx = idx // self.cfg.patch_per_blk
        pch_idx = idx % self.cfg.patch_per_blk
        return int(blk_idx), int(pch_idx)

    def _load_block_to_cache(self, blk_idx: int) -> None:
        '''Load a block into streaming caches if not already present.'''
        # skip loading if block already in the cache
        if blk_idx in self.data.img:
            return
        # otherwise proceed
        blk_name = list(self.blks.keys())[blk_idx] # find blk name by block idx
        blk_data = _BlockDataset(
            self.blks[blk_name],
            self.cfg,
            self._get_domain(blk_name),
            augment_flip=self.aug_flip
        )
        self.data.img[blk_idx] = blk_data.imgs.astype(numpy.float32)
        self.data.lbl[blk_idx] = blk_data.lbls.astype(numpy.int64)
        self.data.dom[blk_idx] = blk_data.domain

    def _get_domain(self, name: str) -> dict[str, int | list[float] | None]:
        '''Retrieve the per-block domain dict if available.'''
        ids: int | None = None
        vec: list[float] | None = None
        if self.cfg.ids_domain:
            ids = self.cfg.ids_domain[name]
        if self.cfg.vec_domain:
            vec = self.cfg.vec_domain[name]
        return {'ids': ids, 'vec': vec}



class _BlockDataset(torch.utils.data.Dataset):
    '''
    Prepare per-patch `(x, y, dom)` samples from a single data block.

    This internal dataset:
      1) Loads a block from `.npz` (expects indexed arrays).
      2) Patchifies the image and label into `(P, C, ps, ps)` and
        `(P, ps, ps)`.
      3) Converts provided per-block domain features into tensors
        (int → long, list[float] → float32) and attaches the same dict
        to each returned patch.
      4) Optionally applies synchronized random horizontal and vertical
        flips to both image and label. Domain tensors are unaffected..

    Attributes:
        config (BlockConfig): Tiling and augmentation configuration.
        domain (dict): Per-block domain dict converted to tensors.
        imgs (numpy.ndarray): Patchified image array of shape
            `(P, C, ps, ps)`.
        lbls (numpy.ndarray): Patchified label array of shape
            `(P, ps, ps)`.
        meta (dict): Metadata loaded from the block (e.g., identifiers).

    Raises:
        ValueError: If patch extraction fails due to incompatible shapes.
    '''

    def __init__(
        self,
        fpath: str,
        config: BlockConfig,
        domains: dict[str, int | list[float] | None],
        *,
        augment_flip: bool = False
    ):
        '''
        Initialize the per-block dataset and patchify arrays.

        Args:
            fpath: Path to the block `.npz` file.
            config:
            domains:

        Raises:
            ValueError: If patch extraction fails due to shape mismatch.
        '''
        super().__init__()

        # process args
        self.config = config
        self.domain: alias.TorchDict = {}
        self.augment_flip = augment_flip

        # load data directly from npz
        loaded = numpy.load(fpath, allow_pickle=True)
        assert config.image_key in loaded and config.label_key in loaded
        try:
            self.imgs = self._get_patches(loaded[config.image_key])
            self.lbls = self._get_patches(loaded[config.label_key])
        except ValueError as err:
            raise ValueError(f'Bad patch at {fpath}') from err

        # parse domain if provided
        for key, dom in domains.items():
            if isinstance(dom, int):
                self.domain[key] = torch.tensor(dom, dtype=torch.long)
            elif isinstance(dom, list):
                self.domain[key] = torch.tensor(dom, dtype=torch.float32)

    def __len__(self) -> int:
        return self.config.patch_per_blk

    def __getitem__(self, idx: int) -> alias.DatasetItem:
        if not idx in range(self.config.patch_per_blk):
            raise KeyError(f'Invalid patch idx: {idx}') # sanity check

        # image should always be valid
        x = torch.from_numpy(self.imgs[idx].astype(numpy.float32))

        # if label is a placeholder array
        if self.lbls.ndim == 1 and self.lbls.shape == (1,):
            y = torch.empty(0, dtype=torch.long)  # passive placeholder
        else:
            y = torch.from_numpy(self.lbls[idx].astype(numpy.int64))
            # augment only when both image and label are present
            if self.augment_flip:
                # one decision for horizontal, one for vertical
                if torch.rand(()) < 0.5:
                    x = torchvision.transforms.functional.hflip(x)
                    y = torchvision.transforms.functional.hflip(y)
                if torch.rand(()) < 0.5:
                    x = torchvision.transforms.functional.vflip(x)
                    y = torchvision.transforms.functional.vflip(y)
        return x, y, self.domain

    def _get_patches(self, arr: numpy.ndarray) -> numpy.ndarray:
        '''Patchify a square block into configured patches.'''
        # unlabelled placeholder, e.g., no-ops label array for image-only data
        # see foundation_data_block.py
        if arr.ndim == 1 and arr.shape == (1,):
            return arr  # keep as-is to signal 'no labels'

        # e.g., arr.shape = [C, 256, 256] ps = 128
        # top row to bottom, within each row left to right
        c, h, w = arr.shape # [C, H, W]
        assert h == w == self.config.block_size # sanity check
        pn = self.config.patch_per_dim # e.g., 2 per dim
        ps = self.config.patch_size # 128
        arr = arr.reshape(c, pn, ps, pn, ps) # [C, 2, 128, 2, 128]
        arr = arr.transpose(1, 3, 0, 2, 4)  # [2, 2, C, 128, 128]
        return arr.reshape(pn * pn, c, ps, ps) # [4, C, 128, 128]


class _CacheDict(collections.OrderedDict):
    '''
    A fixed-size (LRU-like) dictionary for streaming caches.

    Behaves like a normal dict when `maxsize == 0`; otherwise evicts the
    oldest item when inserting a new item that exceeds `maxsize`.

    Attributes:
        maxsize (int): Maximum number of items to retain.
    '''

    def __init__(self, maxsize: int, *args, **kwargs):
        '''
        Initialize the cache dict.

        Args:
            maxsize (int): Maximum number of items; 0 disables eviction.
        '''
        super().__init__(*args, **kwargs)
        self.maxsize = maxsize
        assert isinstance(maxsize, int) and maxsize >= 0

    def __setitem__(self, key, value):
        if self.maxsize == 0:
            super().__setitem__(key, value)
        else:
            # if key exists, move it to the end (newest)
            if key in self:
                del self[key]
            super().__setitem__(key, value)
            # if too many items, remove the oldest
            if len(self) > self.maxsize:
                self.popitem(last=False)  # pop oldest item
