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
Utilities for constructing training/validation/test dataloaders.

This module prepares:
    - Block-based datasets backed by MultiBlockDataset,
    - Mode-specific configurations for train/val/test/single-block use,
    - Optional in-memory preload and caching decisions based on system
      memory,
    - A custom collate function supporting labeled and unlabeled splits,
    - A small metadata bundle shipped with the dataloaders.

The main entry point is `build_dataloaders`, returning a structured
DataLoaders object containing train/val/test loaders and metadata.
'''

# pylint: disable=missing-function-docstring

from __future__ import annotations
# standard imports
import dataclasses
import typing
# third-party imports
import psutil
import torch
import torch.utils.data
# local imports
import landseg.core as core
import landseg.session.common as common
import landseg.session.common.alias as alias
import landseg.session.data as data


# ----- `DataLoaderConfig` protocol
class DataLoaderConfig(typing.Protocol):
    @property
    def batch_size(self) -> int: ...
    @property
    def patch_size(self) -> int: ...


# ----- `DataLoaders` dataclasses
@dataclasses.dataclass
class DataLoaders:
    '''Train/Val/Test dataloader container for the trainer.'''
    train: torch.utils.data.DataLoader | None
    val: torch.utils.data.DataLoader | None
    test: torch.utils.data.DataLoader | None
    meta: _DataLoadersMeta


@dataclasses.dataclass
class _DataLoadersMeta:
    '''Data loaders' metadata.'''
    batch_size: int
    patch_size: int
    preview_context: _PreviewContext | None


@dataclasses.dataclass
class _PreviewContext:
    '''Inference assembly context for block-wise stitching.'''
    patch_per_blk: int
    patch_per_dim: int
    block_columns: int
    patch_grid_shape: tuple[int, int]


@dataclasses.dataclass
class _MemoryFlags:
    '''Dataset memory preload and caching strategy flags.'''
    preload_train: bool
    cache_train: int
    preload_val: bool
    cache_val: int
    preload_test: bool
    cache_test: int


# ----- `build_dataloaders` function
def build_dataloaders(
    data_specs: core.DataSpecs,
    config: DataLoaderConfig,
    *,
    logger: common.SessionLogger | None = None,
) -> DataLoaders:
    '''
    Construct train/val/test dataloaders and metadata from dataset specs.

    Builds PyTorch-style dataloaders based on dataset metadata and user
    configuration, while deriving execution-specific strategies such as
    caching, preloading, and batching behavior.

    Args:
        data_specs: Dataset specification describing data layout, splits,
            block structure, and global metadata.
        config: Data loading configuration (e.g., batch and patch size).
        logger: Optional logger used for reporting build progress.

    Returns:
        DataLoaders: Container with:
            - train: Training dataloader (or None if unavailable)
            - val: Validation dataloader (or None if unavailable)
            - test: Test/inference dataloader (or None if unavailable)
            - meta: Associated metadata (batch/patch settings and
              optional preview context)

    Notes:
        - Loader reuse may occur (e.g., single-block datasets).
        - Memory-aware strategies (e.g., caching/preloading) are inferred
          at runtime.
        - Returned loaders are aligned with dataset structure and
          orchestration requirements.
    '''
    # meta to be shipped
    h_w = data_specs.meta.image_specs.height_width
    assert h_w % config.patch_size == 0

    # return dataloaders container by mode
    match data_specs.mode:

        # if single block mode (for overfit test)
        case 'single':
            single_loader = _load('single', data_specs, config, logger=logger)
            # pass the same as val dataloader for minimal disturbance downstream
            return DataLoaders(
                train=single_loader,
                val=single_loader,
                test=None,
                meta=_DataLoadersMeta(
                    config.batch_size,
                    config.patch_size,
                    preview_context=None
                )
            )

        # val only mode
        case 'val_only':
            val_loader = _load('val', data_specs, config, logger=logger)
            return DataLoaders(
                train=None,
                val=val_loader,
                test=None,
                meta=_DataLoadersMeta(
                    config.batch_size,
                    config.patch_size,
                    preview_context=None
                )
            )

        # test only mode
        case 'test_only':
            test_loader = _load('test', data_specs, config, logger=logger)
            return DataLoaders(
                train=None,
                val=None,
                test=test_loader,
                meta=_DataLoadersMeta(
                    config.batch_size,
                    config.patch_size,
                    preview_context=None
                )
            )

        # default normal experiment
        case 'default':
            train = _load('train', data_specs, config, logger=logger)
            val = _load('val', data_specs, config, logger=logger)
            test = _load('test', data_specs, config, logger=logger)
            if test is not None:
                preview_context = _generate_preview_context(
                    per_blk=int(h_w / config.patch_size) ** 2,
                    test_blks_grid=data_specs.meta.test_blks_grid
                )
            else:
                preview_context = None
            return DataLoaders(
                train=train,
                val=val,
                test=test,
                meta=_DataLoadersMeta(
                    config.batch_size,
                    config.patch_size,
                    preview_context=preview_context
                )
            )


# ----- helper functions
@typing.overload
def _load(
    mode: typing.Literal['single'],
    data_specs: core.DataSpecs,
    config: DataLoaderConfig,
    *,
    logger: common.SessionLogger | None = None,
) -> torch.utils.data.DataLoader: ...


@typing.overload
def _load(
    mode: typing.Literal['train', 'val', 'test'],
    data_specs: core.DataSpecs,
    config: DataLoaderConfig,
    *,
    logger: common.SessionLogger | None = None,
) -> torch.utils.data.DataLoader | None: ...


def _load(
    mode: typing.Literal['train', 'val', 'test', 'single'],
    data_specs: core.DataSpecs,
    config: DataLoaderConfig,
    *,
    logger: common.SessionLogger | None = None,
) -> torch.utils.data.DataLoader | None:
    '''Get a specific dataloader.'''

    # fetch data blocks by mode
    data_blocks_registry: dict[str, dict[str, str]] = {
        'train': data_specs.splits.train,
        'val': data_specs.splits.val,
        'test': data_specs.splits.test,
        'single': data_specs.splits.train
    }
    data_blocks = data_blocks_registry[mode]

    # early exit if no data blocks are available, e.g., no test dataset
    if not data_blocks:
        return None

    # dataset configuration
    dataset_config = _config_by_mode(mode, data_specs, config.patch_size)

    # preload/cache config
    flags = _infer_memory_flags(data_specs)
    preload = getattr(flags, f'preload_{mode}', False)
    cache_num = getattr(flags, f'cache_{mode}', 0)

    # get multiblock dataset
    dataset = data.MultiBlockDataset(
        data_blocks,
        dataset_config,
        preload=preload,
        augment_flip=bool(mode == 'train'),
        blk_cache_num=cache_num
    )

    # log dataloading info if logger is present
    if logger is not None:
        logger.set_inputs({
            mode: {
                'loaded': dataset.n_preloaded,
                'cached': dataset.n_cached
            }
        })
        logger.log(
            'INFO',
            f'Blocks type\t[{mode}]: '
            f'Loaded {dataset.n_preloaded} blocks | '
            f'Cached {dataset.n_cached} blocks'
        )

    # torch dataloader arguments dict
    dataloader_args = {
        'batch_size': 1 if mode == 'single' else config.batch_size,
        'shuffle': mode == 'train',
        'collate_fn': _collate_multi_block
    }
    return torch.utils.data.DataLoader(dataset, **dataloader_args)


def _config_by_mode(
    mode: str,
    data_specs: core.DataSpecs,
    patch_size: int,
) -> data.BlockConfig:
    '''Configure dataloading by mode.'''

    # fetch domain by mode
    domains_registry = {
        'train': data_specs.domains.train,
        'val': data_specs.domains.val,
        'test': data_specs.domains.test,
        'single': None
    }
    domain = domains_registry[mode]

    # return config by mode
    if mode == 'single':
        patch_size = data_specs.meta.image_specs.height_width # single block
    return data.BlockConfig(
        block_size=data_specs.meta.image_specs.height_width,
        patch_size=patch_size,
        image_key=data_specs.meta.image_specs.array_key,
        label_key=data_specs.meta.label_specs.array_key,
        ids_domain=domain['ids_domain'] if domain else None,
        vec_domain=domain['vec_domain'] if domain else None
    )


def _infer_memory_flags(
    data_specs: core.DataSpecs,
    available_bytes: int | None = None
) -> _MemoryFlags:
    '''Infer dataset preload and caching strategy flags based on RAM.'''

    # fit block byte size
    fbytes = data_specs.meta.blk_bytes
    # early exit for single block test (fbytes = 0)
    if not fbytes:
        return _MemoryFlags(
            preload_train=True,
            cache_train=0,
            preload_val=True,
            cache_val=0,
            preload_test=True,
            cache_test=0
        )

    # get dataset sizes
    train_bytes = len(data_specs.splits.train or {}) * fbytes
    val_bytes = len(data_specs.splits.val or {}) * fbytes

    # decision on preload and cache size
    mem = (
        available_bytes
        if available_bytes is not None
        else psutil.virtual_memory().available
    )
    _val = _train = False
    _val_n = train_n = 0

    # first priority: preload validation blocks into memory
    if val_bytes <= 0.6 * mem:
        _val = True
        # second priority: preload training blocks if possible
        if train_bytes <= 0.6 * (mem - val_bytes):
            _train = True
            train_n = round(0.1 * (mem - val_bytes) / fbytes)
    else:
        _val_n = round(0.3 * mem / fbytes)
        train_n = round(0.2 * mem / fbytes)

    # return flags container
    return _MemoryFlags(
        preload_train=_train,
        cache_train=train_n,
        preload_val=_val,
        cache_val=_val_n,
        preload_test=True,
        cache_test=0
    )


def _collate_multi_block(batch: alias.DatasetBatch) -> alias.DatasetItem:
    '''
    Customized collate function to properly stack a batch.

    Contract per split:
      - Labeled: every y is [ps, ps] (long) -> stacked to [B, ps, ps]
      - Unlabeled: every y is empty tensor -> stacked to [B, 0] (long)
      - Domain: all items share the same keys; each stacks to [B, ...]
    '''

    # unpack batch items into separate lists
    xs, ys, ds = zip(*batch)

    # x is always stackable
    xs_out = torch.stack(xs, dim=0) # x -> [B, C, H, W]

    # determine if this is a labeled or unlabeled batch from the first item
    y0 = ys[0]
    labeled_batch = y0.numel() > 0

    if labeled_batch:
        # ensure all y match shape of first y
        exp_shape = y0.shape
        for i, y in enumerate(ys):
            if y.shape != exp_shape:
                raise ValueError(
                    f'inconsistent y shapes in batch at index {i}: '
                    f'expected {tuple(exp_shape)} but got {tuple(y.shape)}'
                )
        ys_out = torch.stack(ys, dim=0).long()
    else:
        # unlabeled/inference: all y must be empty tensors
        for i, y in enumerate(ys):
            if y.numel() != 0:
                raise ValueError(
                    f'mixed labeled/unlabeled batch: item {i} has non-empty y'
                )
        ys_out = torch.stack(ys, dim=0).long()

    # domain assumes consistent keys across batch
    dom_out = {} # -> dict[str, [B, V]] or dict[str, [B]]
    first_dom = ds[0]
    for key in first_dom.keys():
        dom_out[key] = torch.stack([d[key] for d in ds], dim=0)

    return xs_out, ys_out, dom_out


def _generate_preview_context(
    per_blk: int,
    test_blks_grid: tuple[int, int]
) -> _PreviewContext:
    '''Resolve patch-block layout for preview context.'''

    # resolve patch-block layout
    per_dim = int(per_blk ** 0.5)
    assert per_dim * per_dim == per_blk, 'patch_per_blk must be square'

    # resolve block col/row numbers
    blk_col, blk_row = test_blks_grid

    # resolve patch col/row numbers
    pch_col, pch_row = (blk_col * per_dim, blk_row * per_dim)

    return _PreviewContext(
        patch_per_blk=per_blk,
        patch_per_dim=per_dim,
        block_columns=blk_col,
        patch_grid_shape=(pch_col, pch_row)
    )
