'''Get dataloaders'''

# standard imports
from __future__ import annotations
import copy
import dataclasses
# third-party imports
import psutil
import torch
import torch.utils.data
# local imports
import landseg.alias as alias
import landseg.training.common as common
import landseg.training.dataloading as dataloading
import landseg.utils as utils

@dataclasses.dataclass
class DataLoaders:
    '''doc'''
    train: torch.utils.data.DataLoader
    val: torch.utils.data.DataLoader
    test: torch.utils.data.DataLoader | None
    meta: _LoaderMeta

@dataclasses.dataclass
class _LoaderMeta:
    '''Simple meta to be shipped with the dataloaders.'''
    batch_size: int
    patch_per_blk: int
    test_blks_grid:  tuple[int, int]

@dataclasses.dataclass
class _LoadingFlags:
    '''Flags to be consumed during loader creation.'''
    train_preload: bool
    train_cache: int
    val_preload: bool
    val_cache: int

def get_dataloaders(
    data_specs: common.DataSpecsLike,
    loader_config: alias.ConfigType,
    logger: utils.Logger,
) -> DataLoaders:
    '''Entry to the module, returns two dataloaders for training.'''

    # get a child from the base logger
    logger = logger.get_child('dldrs')

    # parse args from config accessor
    loader_cfg = utils.ConfigAccess(loader_config)

    # get dataset filepaths from DataSummary
    data_paths = data_specs.splits
    domains = data_specs.domains

    # get loading flags
    flags = _get_flags(data_specs)

    # declare loaders type and defualt value
    train_loader: torch.utils.data.DataLoader
    val_loader: torch.utils.data.DataLoader
    test_loader: torch.utils.data.DataLoader | None = None

    # get persistent block config
    _cfg = dataloading.BlockConfig(
        loader_cfg.get_option('block_size'),
        loader_cfg.get_option('patch_size'),
    )
    batch_size = loader_cfg.get_option('batch_size')
    # meta to be shipped
    meta = _LoaderMeta(
        batch_size,
        _cfg.patch_per_blk,
        data_specs.meta.test_blks_grid
    )

    # training loader
    # config
    cfg = copy.deepcopy(_cfg) # avoid contamination from other loaders
    cfg.augment_flip = True
    cfg.ids_domain = domains.train['ids_domain']
    cfg.vec_domain = domains.train['vec_domain']
    # get dataset
    dataset = dataloading.MultiBlockDataset(
        data_paths.train,
        cfg,
        logger,
        preload=flags.train_preload,
        blk_cache_num=flags.train_cache
    )
    # get dataloader
    train_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=_collate_multi_block
    )

    # validation loader
    # config
    cfg = copy.deepcopy(_cfg) # avoid contamination from other loaders
    cfg.augment_flip = False
    cfg.ids_domain = domains.val['ids_domain']
    cfg.vec_domain = domains.val['vec_domain']
    # get dataset
    dataset = dataloading.MultiBlockDataset(
        data_paths.val,
        cfg,
        logger,
        preload=flags.val_preload,
        blk_cache_num=flags.val_cache
    )
    # get dataloader
    val_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=_collate_multi_block
    )

    # test loader if provided
    if data_paths.test:
        # config
        cfg = copy.deepcopy(_cfg) # avoid contamination from other loaders
        cfg.augment_flip = False
        cfg.ids_domain = domains.test['ids_domain']
        cfg.vec_domain = domains.test['vec_domain']
        # get dataset
        dataset = dataloading.MultiBlockDataset(
            data_paths.test,
            cfg,
            logger,
            preload=True,   # assuming small dataset, always preload
        )
        # get dataloader
        test_loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=_collate_multi_block
        )

    # return
    return DataLoaders(train_loader, val_loader, test_loader, meta)

def _get_flags(data_summary: common.DataSpecsLike) -> _LoadingFlags:
    '''Get dataset loading flags.'''

    # get dataset filepaths from DataSummary
    data = data_summary.splits
    fbytes = data_summary.meta.fit_perblk_bytes     # fit block byte size

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
    return _LoadingFlags(_train, train_n, _val, _val_n)

def _collate_multi_block(batch: alias.DatasetBatch) -> alias.DatasetItem:
    '''
    Customized collate function to properly stack a batch.

    Contract per split:
      - Labeled split: every y is [ps, ps] (long) -> stacked to [B, ps, ps]
      - Unlabeled split: every y is empty tensor -> stacked to [B, 0] (long)
      - Domain: all items share the same keys; each value stacks to [B, ...]
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
