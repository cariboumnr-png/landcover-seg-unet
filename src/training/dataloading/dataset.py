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
import tqdm
# local imports
import dataset
import utils

@dataclasses.dataclass
class BlockConfig:
    '''
    Simple dataclass to hold block processing config.

    This dataclass holds the parameters that control how an image block
    is split into uniform patches and whether simple augmentations
    (horizontal/vertical flips) are applied. It can also carry optional
    per-block domain features - categorical IDs or continuous covariate
    vectors to be attached to each returned sample.

    Attributes:
        augment_flip (bool): Whether to apply random horizontal/vertical
            flips to both image and label tensors during sampling.
        block_size (int): Size (pxs) of the square input block (H == W).
        patch_size (int): Size (pxs) of each square patch extracted.
        domain_map (dict | None): Optional mapping from a block ID to a
            domain feature dictionary. Each dict value should be either
            an `int` (categorical ID) or a `list[float]` (vector of
            continuous features).
        patch_per_dim (int): Derived number of patches per spatial
            dimension, set in `__post_init__`.
        patch_per_block (int): Derived total patches per block, set in
            `__post_init__`.

    Raises:
        AssertionError: If `block_size < patch_size` or `block_size` is
            not divisible by `patch_size`.
    '''

    block_size: int
    patch_size: int
    augment_flip: bool = False
    domain_dict: dict[str, dict[str, int | list[float]]] | None = None
    patch_per_dim: int = dataclasses.field(init=False)
    patch_per_blk: int = dataclasses.field(init=False)

    def __post_init__(self):
        assert self.block_size >= self.patch_size
        assert self.block_size % self.patch_size == 0
        self.patch_per_dim = int(self.block_size // self.patch_size)
        self.patch_per_blk = self.patch_per_dim ** 2

    def __repr__(self):
        return repr(dataclasses.asdict(self))

@dataclasses.dataclass
class _MultiBlockData:
    '''Small container for multiblock data.'''
    img: numpy.ndarray | _CacheDict = dataclasses.field(init=False)
    lbl: numpy.ndarray | _CacheDict = dataclasses.field(init=False)
    dom: list[dict[str, torch.Tensor]] | _CacheDict = dataclasses.field(init=False)

class MultiBlockDataset(torch.utils.data.Dataset):
    '''
    Unified dataset over multiple block .npz files with patch extraction
    and optional domain features.

    This dataset abstracts a set of block files and yields fixed-size
    patches `(x, y, dom)` for training segmentation models.

    Operating modes:
    * `preload=True`: eagerly loads all blocks, patchifies, and keeps
        arrays in memory. Ideal for validation (static across epochs).

    * `preload=False`: lazy mapping from a global index to (block, local
        patch), loads only the needed block into a small cache. Ideal
        for large training sets.

    The optional `domain_map` in `BlockConfig` is consulted per block;
    when present, a per-block domain dict is attached to each returned
    patch. Integer values are converted to `torch.long` and list values
    to `torch.float32`.

    Attributes:
        fpaths (list[str]): List of block file paths (`.npz`).
        preload (bool): Whether to preload or stream.
        block_config (BlockConfig): Block config see the dataclass.
        _imgs: Storage for image patches. `list[ndarray]` if preload,
            `_CacheDict` if streaming.
        _lbls: Storage for label patches. `list[ndarray]` if preload,
            `_CacheDict` if streaming.
        _doms: Storage for domain. `list[dict]` if preload, `_CacheDict`
            if streaming.
        _len (int): Total number of patches across all blocks.

    Raises:
        ValueError: If patch extraction fails for a block due to shape
            or data issues.
    '''

    def __init__(
            self,
            blks_dict: dict[str, str],
            blk_cfg: BlockConfig,
            logger: utils.Logger,
            **kwargs
        ):
        '''
        Initialize the dataset over multiple data blocks (.npz files).

        Args:
            fpaths (dict[str, str]): Block name: paths to `.npz` files.
            blk_cfg (BlockConfig): Tiling, augmentation & domain config.
            preload (bool): If `True`, load and patchify all blocks into
                RAM; else stream with an LRU-like cache.
            blk_cache_num (int): Max number of blocks to keep in cache.
        '''

        # from parent class
        super().__init__()

        # process args
        self.blks = blks_dict
        self.blk_cfg = blk_cfg
        self.logger = logger
        self.preload = kwargs.get('preload', False)
        blk_cache_num = kwargs.get('blk_cache_num', 16)

        # init data container
        self.data = _MultiBlockData()
        # load all files into one dataset
        if self.preload:
            self.logger.log('INFO', 'Preloading blocks into RAM')
            self.logger.log('DEBUG', f'Config: {blk_cfg}')
            _imgs = []
            _lbls = []
            self.data.dom = []
            for blk_name, blk_fpath in tqdm.tqdm(blks_dict.items(), ncols=100):
                dom = self._get_domain(blk_name)
                blk_data = _BlockDataset(blk_fpath, blk_cfg, self.logger, dom)
                _imgs.append(blk_data.imgs)
                _lbls.append(blk_data.lbls)
                self.data.dom.extend([blk_data.domain] * blk_cfg.patch_per_blk)
            # concatenate
            self.data.img = numpy.concatenate(_imgs, axis=0)
            self.data.lbl = numpy.concatenate(_lbls, axis=0)
            self._len = int( self.data.img.shape[0])
            self.logger.log('INFO', f'{len(blks_dict)} blocks preloaded into RAM')

        # otherwise streaming
        else:
            self.logger.log('INFO', 'Setting up block streaming')
            self.logger.log('DEBUG', f'Config: {blk_cfg}')
            self._len = self.blk_cfg.patch_per_blk * len(blks_dict)
            # below not needed for uniform n
            # n =  self.block_config.patch_per_block
            # self.cumulative_i = [n * i for i in range(len(fpaths) + 1)]
            self.data.img = _CacheDict(maxsize=blk_cache_num)
            self.data.lbl = _CacheDict(maxsize=blk_cache_num)
            self.data.dom = _CacheDict(maxsize=blk_cache_num)
            self.logger.log('INFO', f'Streaming cache: {blk_cache_num} blocks')

    def __len__(self):
        return self._len

    def __getitem__(
            self,
            idx: int
        ) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        if self.preload:
            x = self.data.img[idx].astype(numpy.float32)  # [C, ps, ps]
            y = self.data.lbl[idx].astype(numpy.int64)  # [ps, ps]
            dom = self.data.dom[idx] # dict[str, tensor]
            dom = {k: v.clone() for k, v in dom.items()} # new - no shared refs
        else:
            blk_idx, pch_idx = self._global_to_local(idx)
            self._load_block_to_cache(blk_idx)
            x = self.data.img[blk_idx][pch_idx]
            y = self.data.lbl[blk_idx][pch_idx]
            dom = self.data.dom[blk_idx] # domain at is batch-level
            dom = {k: v.clone() for k, v in dom.items()} # new - no shared refs
        return torch.from_numpy(x), torch.from_numpy(y).long(), dom

    def _global_to_local(self, idx: int) -> tuple[int, int]:
        '''Map global patch to block/patch indices (uniform blocks).'''

        blk_idx = idx // self.blk_cfg.patch_per_blk
        pch_idx = idx % self.blk_cfg.patch_per_blk
        return int(blk_idx), int(pch_idx)

    def _load_block_to_cache(self, blk_idx: int) -> None:
        '''Load a block into streaming caches if not already present.'''

        # skip loading if block already in the cache
        if blk_idx in self.data.img:
            return
        # otherwise proceed
        blk_name = list(self.blks.keys())[blk_idx] # find blk name by block idx
        blk_fpath = self.blks[blk_name]
        dom = self._get_domain(blk_name)
        blk_data = _BlockDataset(blk_fpath, self.blk_cfg, self.logger, dom)
        self.data.img[blk_idx] = blk_data.imgs.astype(numpy.float32)
        self.data.lbl[blk_idx] = blk_data.lbls.astype(numpy.int64)
        self.data.dom[blk_idx] = blk_data.domain

    def _get_domain(self, blk_name: str) -> dict[str, int | list[float]] | None:
        '''Retrieve the per-block domain dict if available.'''

        if self.blk_cfg.domain_dict is not None:
            blk_domain = self.blk_cfg.domain_dict.get(blk_name)
            # relaxed check allowing block with empty or missing domain
            if not isinstance(blk_domain, dict) or len(blk_domain) == 0:
                self.logger.log('WARNING', f'Invalid domain for {blk_name}')
                return None
            return blk_domain
        return None

# internal pieces
class _BlockDataset(torch.utils.data.Dataset):
    '''
    Prepare per-patch `(x, y, dom)` samples from a single data block.

    This internal dataset:
      1) Loads a block from `.npz` (image/label/meta).
      2) Patchifies the image and label into `(P, C, ps, ps)` and
        `(P, ps, ps)`.
      3) Converts provided per-block domain features into tensors
        (int → long, list[float] → float32) and attaches the same dict
        to each returned patch.
      4) Optionally applies random horizontal/vertical flips to both
        image and label.

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
            block_fpath: str,
            block_config: BlockConfig,
            logger: utils.Logger,
            block_domain: dict[str, int | list[float]] | None=None
        ):
        '''
        Initialize the per-block dataset and patchify arrays.

        Args:
            block_fpath (str): Path to the block `.npz` file.
            block_config (BlockConfig): Tiling and augmentation
                configuration.
            block_domain (dict | None): Optional domain dict; integer
                values are converted to `torch.long` and list-of-floats
                to `torch.float32`. When absent, `domain` dict is empty.

        Raises:
            ValueError: If patch extraction fails due to shape mismatch.
        '''

        # from parent class
        super().__init__()

        # process args
        self.config = block_config
        self.domain: dict[str, torch.Tensor] = {}
        self.logger = logger

        # load data from npz
        bb = dataset.DataBlock().load(block_fpath)
        self.meta = bb.meta # get metadata of the block for later
        try:
            self.imgs = self._get_patches(bb.data.image_normalized)
            self.lbls = self._get_patches(bb.data.label_masked)
        except ValueError:
            self.logger.log('ERROR', f'Bad patch at {block_fpath}')
            self.logger.log('ERROR', f'Meta:\n{self.meta}')
            raise

        # parse domain if provided
        if block_domain is not None:
            for key, dom in block_domain.items():
                if isinstance(dom, int):
                    self.domain[key] = torch.tensor(dom, dtype=torch.long)
                elif isinstance(dom, list):
                    self.domain[key] = torch.tensor(dom, dtype=torch.float32)

    def __len__(self) -> int:
        return self.config.patch_per_blk

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, dict]:
        assert idx in range(self.config.patch_per_blk) # sanity check
        x = self.imgs[idx].astype(numpy.float32)
        y = self.lbls[idx].astype(numpy.int64)
        x_tensor = torch.from_numpy(x)
        y_tensor = torch.from_numpy(y)
        if self.config.augment_flip:
            # one decision for horizontal, one for vertical
            if torch.rand(()) < 0.5:
                x_tensor = torchvision.transforms.functional.hflip(x_tensor)
                y_tensor = torchvision.transforms.functional.hflip(y_tensor)
            if torch.rand(()) < 0.5:
                x_tensor = torchvision.transforms.functional.vflip(x_tensor)
                y_tensor = torchvision.transforms.functional.vflip(y_tensor)
        return x_tensor, y_tensor, self.domain

    def _get_patches(self, arr: numpy.ndarray) -> numpy.ndarray:
        '''Patchify a square block into configured patches.'''

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
