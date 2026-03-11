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

The main entry point is `get_dataloaders`, returning a structured
DataLoaders object containing train/val/test loaders and metadata.
'''

# standard imports
from __future__ import annotations
import dataclasses
import functools
# third-party imports
import psutil
import torch
import torch.utils.data
# local imports
import landseg.alias as alias
import landseg.configs as configs
import landseg.core as core
import landseg.training.dataloading as dataloading
import landseg.utils as utils

# ------------------------------Public  Dataclass------------------------------
@dataclasses.dataclass
class DataLoaders:
    '''Train/Val/Test dataloader for the trainer.'''
    train: torch.utils.data.DataLoader
    val: torch.utils.data.DataLoader
    test: torch.utils.data.DataLoader | None
    meta: _Meta

# ------------------------------private dataclass------------------------------
@dataclasses.dataclass
class _Meta:
    '''Simple meta to be shipped with the dataloaders.'''
    batch_size: int
    patch_per_blk: int
    test_blks_grid:  tuple[int, int]

# -------------------------------Public Function-------------------------------
def get_dataloaders(
    data_specs: core.DataSpecs,
    config: configs.LoaderConfig,
    logger: utils.Logger,
) -> DataLoaders:
    '''
    Build dataloaders based on dataset metadata and configuration.

    Args:
        data_specs: Dataset specification containing block paths, domain
            information, split definitions, and global metadata.
        loader_config: Configuration describing batch size, block/patch
            size, and dataloading behaviour (shuffle, caching, etc.).
        logger: Base logger used to create a child logger for logging.

    Returns:
        A `DataLoaders` instance containing:
            - train: DataLoader for training blocks,
            - val:   DataLoader for validation blocks,
            - test:  DataLoader for test blocks (or None),
            - meta:  Small metadata bundle (see `_Meta`).

    Notes:
        - In single-block mode, a single loader is built and reused for
          both train and val.
        - Preload and caching decisions are computed dynamically based
          on available system memory.
    '''

    # get a child from the base logger
    logger = logger.get_child('dldrs')

    # parse
    patch_size = config.patch_size
    assert data_specs.meta.img_h_w % patch_size == 0 # sanity check
    patch_per_blk = int(data_specs.meta.img_h_w / patch_size) ** 2
    batch_size = config.batch_size

    # meta to be shipped
    meta = _Meta(batch_size, patch_per_blk, data_specs.meta.test_blks_grid)

    # partial load function
    load_partial = functools.partial(
        _load,
        batch_size=batch_size,
        patch_size=patch_size,
        data_specs=data_specs,
        logger=logger
    )

    # if single block mode (for overfit test)
    if data_specs.mode == 'single':
        single_loader = load_partial('single')
        # pass the same as val dataloader for minimal disturbance downstream
        assert single_loader
        return DataLoaders(single_loader, single_loader, None, meta)

    # otherwise (normal experiment)
    train_loader = load_partial('train')
    val_loader = load_partial('val')
    test_loader = load_partial('test')
    # sanity checks and return
    assert train_loader and val_loader
    return DataLoaders(train_loader, val_loader, test_loader, meta)

# ------------------------------private  function------------------------------
def _load(
    mode: str,
    batch_size: int,
    patch_size: int,
    data_specs: core.DataSpecs,
    logger: utils.Logger
) -> torch.utils.data.DataLoader | None:
    '''Get a specific dataloader.'''

    # mode sanity check
    assert mode in ['train', 'val', 'test', 'single']

    # fetch data blocks by mode
    data_blocks_registry = {
        'train': data_specs.splits.train,
        'val': data_specs.splits.val,
        'test': data_specs.splits.test,
        'single': data_specs.splits.train   # the single block will be here
    }
    data_blocks = data_blocks_registry[mode]
    # early exit if no data blocks are available, e.g., no test dataset
    if not data_blocks:
        return None
    # dataset configuration
    dataset_config = _config_by_mode(mode, patch_size, data_specs)
    # preload/cache config
    load_options = _preload_option(data_specs)
    # get multiblock dataset
    dataset = dataloading.MultiBlockDataset(
        data_blocks,
        dataset_config,
        logger,
        preload=load_options[f'preload_{mode}'],
        augment_flip=bool(mode == 'train'),
        blk_cache_num=load_options[f'cache_{mode}']
    )

    # get dataloader
    # torch dataloader arguments dict
    dataloader_args =  {
        'batch_size': 1 if mode == 'single' else batch_size,
        'shuffle': mode == 'train',
        'collate_fn': _collate_multi_block
    }
    # return dataloader
    return torch.utils.data.DataLoader(dataset, **dataloader_args)

def _config_by_mode(
    mode: str,
    patch_size: int,
    data_specs: core.DataSpecs
):
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
        patch_size = data_specs.meta.img_h_w # single block
    return dataloading.BlockConfig(
        block_size=data_specs.meta.img_h_w,
        patch_size=patch_size,
        array_keys = {
            'image_key': data_specs.meta.img_arr_key,
            'label_key': data_specs.meta.lbl_arr_key
        },
        ids_domain=domain['ids_domain'] if domain else None,
        vec_domain=domain['vec_domain'] if domain else None
    )

def _preload_option(data_specs: core.DataSpecs) -> dict[str, int | bool]:
    '''Get dataset loading flags.'''

    # get dataset filepaths from DataSummary
    data = data_specs.splits
    fbytes = data_specs.meta.fit_perblk_bytes     # fit block byte size

    # early exit for single block test (fbytes = 0)
    if not fbytes:
        return {
            'preload_single': True,
            'cache_single': 0
        }

    # get dataset sizes
    train_bytes = len(data.train or {}) * fbytes
    val_bytes = len(data.val or {}) * fbytes

    # decision on preload and cache size
    mem = psutil.virtual_memory().available
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

    # return flags
    return {
        'preload_train': _train,
        'cache_train': train_n,
        'preload_val': _val,
        'cache_val': _val_n,
        'preload_test': True,
        'cache_test': 0
    }

def _collate_multi_block(batch: alias.DatasetBatch) -> alias.DatasetItem:
    '''
    Customized collate function to properly stack a batch.

    Contract per split:
      - Labeled: every y is [ps, ps] (long) -> stacked to [B, ps, ps]
      - Unlabeled: every y is empty tensor -> stacked to [B, 0] (long)
      - Domain: all items share the same keys; each stacks to [B, ...]
    '''

    # unpack batch as a list
    xs, ys, ds = zip(*batch)            # length B:
    xs = [x for x, _, _ in batch]       # list[Tensor]
    ys = [y for _, y, _ in batch]       # list[Tensor]
    ds = [d for _, _, d in batch]       # list[TorchDict]

    # x is always stackable
    xs = torch.stack(xs, dim=0) # x -> [B, C, H, W]

    # y can be labeled or unlabeled - fail fast if mixed
    # determine if this is a labeled or unlabeled batch from the first item
    y0 = ys[0] # read first and determined. assuming homogeny
    labeled_batch = y0.numel() > 0
    # if labeled/training
    if labeled_batch:
        # Guard: ensure all y match shape of first_y
        exp_shape = y0.shape
        for i, y in enumerate(ys):
            if y.shape != exp_shape:
                raise ValueError(
                    f'inconsistent y shapes in batch at index {i}: '
                    f'expected {tuple(exp_shape)} but got {tuple(y.shape)}'
                )
        ys_out = torch.stack(ys, dim=0).long()
    # unlabeled/inference: all y must be empty tensors
    else:
        for i, y in enumerate(ys):
            if y.numel() != 0:
                raise ValueError(
                    f'mixed labeled/unlabeled batch: item {i} has non-empty y'
                )
        # Stack to [B, 0]; torch.stack works for same shape zero-length tensors
        ys_out = torch.stack(ys, dim=0).long()

    # domain assumes consistent keys across batch
    dom_out = {} # -> dict[str, [B, V]] or dict[str, [B]]
    first_dom = ds[0]
    for key in first_dom.keys():
        dom_out[key] = torch.stack([d[key] for d in ds], dim=0)

    # return
    return xs, ys_out, dom_out
