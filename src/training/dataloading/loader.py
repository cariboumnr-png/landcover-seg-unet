'''Get dataloaders'''

# standard imports
import dataclasses
# third-party imports
import psutil
import torch
import torch.utils.data
# local imports
import _types
import training.common
import training.dataloading
import utils

@dataclasses.dataclass
class DataLoaders:
    '''doc'''
    train: torch.utils.data.DataLoader | None
    val: torch.utils.data.DataLoader | None
    infer: torch.utils.data.DataLoader | None

    def __post_init__(self):
        if not self.train and not self.val:
            assert self.infer
        if not self.infer:
            assert self.train and self.val

@dataclasses.dataclass
class _LoadingFlags:
    '''Flags to be consumed during loader creation.'''
    train_preload: bool
    val_preload: bool
    infer_preload: bool
    train_cache: int
    val_cache: int
    infer_cache: int

def get_dataloaders(
        mode: str,
        data_summary: training.common.DataSummaryLike,
        loader_config: _types.ConfigType,
        logger: utils.Logger,
    ) -> DataLoaders:
    '''Entry to the module, returns two dataloaders for training.'''

    # get a child from the base logger
    logger = logger.get_child('dldrs')

    # parse args from config accessor
    loader_cfg = utils.ConfigAccess(loader_config)

    # get dataset filepaths from DataSummary
    data_paths = data_summary.data
    domain_paths = data_summary.doms

    # get loading flags
    flags = _get_flags(data_summary)

    # declare loaders type and defualt value
    t_loader: torch.utils.data.DataLoader | None = None
    v_loader: torch.utils.data.DataLoader | None = None
    i_loader: torch.utils.data.DataLoader | None = None

    # get persistent block config
    cfg = training.dataloading.BlockConfig(
        block_size=loader_cfg.get_option('block_size'),
        patch_size=loader_cfg.get_option('patch_size'),
    )
    batch_size = loader_cfg.get_option('batch_size')

    # by work mode
    if mode in ['with_inference', 'no_inference']:
        assert data_paths.train and data_paths.val

        # training loader
        # config
        cfg.augment_flip = True
        cfg.domain_dict = domain_paths.train_val_domain
        # data
        data = training.dataloading.MultiBlockDataset(
            blks_dict=data_paths.train,
            blk_cfg=cfg,
            logger=logger,
            preload=flags.train_preload,
            blk_cache_num=flags.train_cache
        )
        # loader
        t_loader = torch.utils.data.DataLoader(
            dataset=data,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=_collate_multi_block
        )

        # validation loader
        # config
        cfg.augment_flip = False
        cfg.domain_dict = domain_paths.train_val_domain
        # data
        data = training.dataloading.MultiBlockDataset(
            blks_dict=data_paths.val,
            blk_cfg=cfg,
            logger=logger,
            preload=flags.val_preload,
            blk_cache_num=flags.val_cache
        )
        # loader
        v_loader = torch.utils.data.DataLoader(
            dataset=data,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=_collate_multi_block
        )

        if mode == 'with_inference':

            assert data_paths.infer
            # inference loader
            # config
            cfg.augment_flip = False
            cfg.domain_dict = domain_paths.infer_domain
            # data
            data = training.dataloading.MultiBlockDataset(
                blks_dict=data_paths.infer,
                blk_cfg=cfg,
                logger=logger,
                preload=flags.infer_preload,
                blk_cache_num=flags.infer_cache
            )
            # loader
            i_loader = torch.utils.data.DataLoader(
                dataset=data,
                batch_size=batch_size,
                shuffle=False,
                collate_fn=_collate_multi_block
            )
    elif mode == 'inference_only':

        assert data_paths.infer
        # inference loader
        # config
        cfg.augment_flip = False
        cfg.domain_dict = domain_paths.infer_domain
        # data
        data = training.dataloading.MultiBlockDataset(
            blks_dict=data_paths.infer,
            blk_cfg=cfg,
            logger=logger,
            preload=flags.infer_preload,
            blk_cache_num=flags.infer_cache
        )
        # loader
        i_loader = torch.utils.data.DataLoader(
            dataset=data,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=_collate_multi_block
        )

    else:
        raise ValueError(f'Invalide mode: {mode}')

    # final sanity
    # at least one loader is not None
    assert any([t_loader, v_loader, i_loader])
    # training and validation loaders are tied together
    assert (t_loader and v_loader) or not (t_loader and not v_loader)
    # return
    return DataLoaders(t_loader, v_loader, i_loader)

# def _parse_training_loader(
#         cfg: training.dataloading.BlockConfig,
#         data_summary: training.common.DataSummaryLike,
#         logger: utils.Logger,
#         flags: _LoadingFlags,
#         batch_size: int
#     ) -> torch.utils.data.DataLoader:
#     '''doc'''

#     assert data_summary.data.train is not None
#     cfg.augment_flip = False
#     cfg.domain_dict = data_summary.doms.train_val_domain
#     # data
#     data = training.dataloading.MultiBlockDataset(
#         blks_dict=data_summary.data.train,
#         blk_cfg=cfg,
#         logger=logger,
#         preload=flags.val_preload,
#         blk_cache_num=flags.val_cache
#     )
#     # loader
#     loader = torch.utils.data.DataLoader(
#         dataset=data,
#         batch_size=batch_size,
#         shuffle=False,
#         collate_fn=_collate_multi_block
#     )
#     return loader

def _get_flags(data_summary: training.common.DataSummaryLike) -> _LoadingFlags:
    '''Get flags.'''

    # get dataset filepaths from DataSummary
    data = data_summary.data
    t_v_bytes = data_summary.meta.train_val_blk_bytes
    i_bytes = data_summary.meta.infer_blk_bytes

    # get dataset sizes
    train_bytes = len(data.train or {}) * t_v_bytes
    val_bytes = len(data.val or {}) * t_v_bytes
    infer_bytes = len(data.infer or {}) * i_bytes

    # decision on preload and cache size
    mem = psutil.virtual_memory().available
    v_pre = t_pre = i_pre = False
    v_cac = t_cac = i_cac = 0
    # first priority: preload validation data into memory
    if val_bytes <= 0.6 * mem:
        v_pre = True
        # second priority: preload training data if possible
        if train_bytes <= 0.6 * (mem - val_bytes):
            t_pre = True
            # last priority: preload inference data if possible
            if infer_bytes <= 0.6 * (mem - val_bytes - train_bytes):
                i_pre = True
            else:
                i_cac = round(0.1 * (mem - val_bytes - train_bytes) / i_bytes)
        else:
            t_cac = round(0.1 * (mem - val_bytes) / t_v_bytes)
    else:
        v_cac = round(0.3 * mem / t_v_bytes)
        t_cac = round(0.2 * mem / t_v_bytes)
        i_cac = round(0.1 * mem / data_summary.meta.infer_blk_bytes)

    # return flags
    return _LoadingFlags(t_pre, v_pre, i_pre, t_cac, v_cac, i_cac)

def _collate_multi_block(batch: _types.DatasetBatch) -> _types.DatasetItem:
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
