'''Get dataloaders'''

# standard imports
from __future__ import annotations
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

# ------------------------------Public  Dataclass------------------------------
@dataclasses.dataclass
class DataLoaders:
    '''Train/Val/Test dataloader for the trainer.'''
    train: torch.utils.data.DataLoader
    val: torch.utils.data.DataLoader
    test: torch.utils.data.DataLoader | None
    meta: _LoaderMeta

# ------------------------------private dataclass------------------------------
@dataclasses.dataclass
class _LoaderMeta:
    '''Simple meta to be shipped with the dataloaders.'''
    batch_size: int
    patch_per_blk: int
    test_blks_grid:  tuple[int, int]

# -------------------------------Public Function-------------------------------
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

    # meta to be shipped
    meta = _LoaderMeta(
        loader_cfg.get_option('batch_size'),
        loader_cfg.get_option('patch_size'),
        data_specs.meta.test_blks_grid
    )

    # if single block mode (for overfit test)
    if data_specs.meta.single_block_mode:
        single_loader = _load('single', loader_cfg, data_specs, logger)
        # pass the same as val dataloader for minimal disturbance downstream
        assert single_loader
        return DataLoaders(single_loader, single_loader, None, meta)

    # otherwise (normal experiment)
    train_loader = _load('train', loader_cfg, data_specs, logger)
    val_loader = _load('val', loader_cfg, data_specs, logger)
    test_loader = _load('test', loader_cfg, data_specs, logger)
    # sanity checks and return
    assert train_loader and val_loader
    return DataLoaders(train_loader, val_loader, test_loader, meta)

# ------------------------------private  function------------------------------
def _load(
    mode: str,
    loader_cfg: utils.ConfigAccess,
    data_specs: common.DataSpecsLike,
    logger: utils.Logger
) -> torch.utils.data.DataLoader | None:
    '''Get a specific dataloader.'''

    # mode sanity check
    assert mode in ['train', 'val', 'test', 'single']

    # prepare dataset by mode
    # dataset path registry
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
    dataset_config = _mode_configurator(mode, loader_cfg, data_specs)
    # preload/cache config
    load_options = _preload_option(data_specs)
    dataset = dataloading.MultiBlockDataset(
        data_blocks,
        dataset_config,
        logger,
        preload=load_options[f'preload_{mode}'],
        blk_cache_num=load_options[f'cache_{mode}']
    )

    # get dataloader
    # torch dataloader arguments dict
    batch_size = loader_cfg.get_option('batch_size')
    dataloader_args =  {
        'batch_size': 1 if mode == 'single' else batch_size,
        'shuffle': mode == 'train',
        'collate_fn': _collate_multi_block
    }
    # return dataloader
    return torch.utils.data.DataLoader(dataset, **dataloader_args)

def _mode_configurator(
    mode: str,
    loader_cfg: utils.ConfigAccess,
    data_specs: common.DataSpecsLike
):
    '''Configure dataloading by mode.'''

    # domain registry
    domains_registry = {
        'train': data_specs.domains.train,
        'val': data_specs.domains.val,
        'test': data_specs.domains.test,
        'single': None
    }
    domain = domains_registry[mode]

    # return config by mode
    block_size = loader_cfg.get_option('block_size')
    if mode == 'single':
        patch_size = block_size # single block
    else:
        patch_size = loader_cfg.get_option('patch_size')
    return dataloading.BlockConfig(
        block_size,
        patch_size,
        mode == 'train',
        domain['ids_domain'] if domain else None,
        domain['vec_domain'] if domain else None
    )

def _preload_option(data_specs: common.DataSpecsLike) -> dict[str, int | bool]:
    '''Get dataset loading flags.'''

    # get dataset filepaths from DataSummary
    data = data_specs.splits
    fbytes = data_specs.meta.fit_perblk_bytes     # fit block byte size

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
        'cache_val': _val_n
    }

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
