'''Summarize the existing cache blocks and generate metadata.'''

# standard imports
from __future__ import annotations
import dataclasses
import math
import os
import random
# local imports
import alias
import dataset
import utils

# ------------------------------Public  Dataclass------------------------------
@dataclasses.dataclass
class DataSummary:
    '''doc'''
    meta: _Meta
    heads: _Heads
    data: _Data
    doms: _Domain

    def __post_init__(self):
        # either train+val or train+val+infer or infer only
        if not self.data.train and not self.data.val:
            assert self.data.infer
        if not self.data.infer:
            assert self.data.train and self.data.val
        # heads match with logit adjustment
        assert self.heads.class_counts.keys() == self.heads.logits_adjust.keys()
        # if domain is provided, all blocks should have it
        if self.data.train and self.data.val and self.doms.fit_domain:
            assert len(self.data.train) + len(self.data.val) == len(self.doms.fit_domain)

    def __str__(self) -> str:
        def _ln(lst):
            return [round(x, 4) for x in lst]
        t1 = '\n - '.join([
            f'{k}:\t{v}' for k, v in self.heads.class_counts.items()
        ])
        t2 = '\n - '.join([
            f'{k}:\t{_ln(v)}' for k, v in self.heads.logits_adjust.items()
        ])
        return '\n'.join([
            'Dataset summary:\n----------------------------------------',
            f'Dataset name: {self.meta.dataset_name}'
            f'Ignore index: {self.meta.ignore_index}',
            f'Number of image channels: {self.meta.img_ch_num}',
            f'Class distribution of each head: \n - {t1}',
            f'Head-wise logits adjustment: \n - {t2}',
            f'Number of training blocks: {len(self.data.train or {})}',
            f'Number of validation blocks: {len(self.data.val or {})}',
            f'Number of inference blocks: {len(self.data.val or {})}',
            f'Domain knowledge includes: {self.doms.fit_meta}'
        ])

# ------------------------------private dataclass------------------------------
@dataclasses.dataclass
class _Meta:
    '''Metadata.'''
    dataset_name: str
    fit_blks_bytes: int
    test_blk_bytes: int
    ignore_index: int
    img_ch_num: int

@dataclasses.dataclass
class _Heads:
    '''Heads label stats'''
    class_counts: dict[str, list[int]]
    logits_adjust: dict[str, list[float]]
    topology: dict[str, dict[str, str | int | None]]

@dataclasses.dataclass
class _Data:
    '''Block file paths of training and validation datasets.'''
    train: dict[str, str] | None
    val: dict[str, str] | None
    infer: dict[str, str] | None

@dataclasses.dataclass
class _Domain:
    '''Domain related.'''
    fit_domain: dict[str, dict[str, int | list[float]]] | None
    fit_meta: dict[str, dict[str, str | int | list]] | None
    test_domain: dict[str, dict[str, int | list[float]]] | None
    test_meta: dict[str, dict[str, str | int | list]] | None

# -------------------------------Public Function-------------------------------
def generate_summary(
    dataset_name: str,
    cache_config: alias.ConfigType
) -> DataSummary:
    '''Wrapper function to generate concrete `DataSummary` class.'''

    # config accessor
    cache_cfg = utils.ConfigAccess(cache_config)

    # get artifacts names
    lbl_count = cache_cfg.get_asset('artifacts', 'label', 'count_training')
    square_blks = cache_cfg.get_asset('artifacts', 'blocks', 'square')
    train_val_blks = cache_cfg.get_asset('artifacts', 'blocks', 'valid')
    training_blks = cache_cfg.get_asset('artifacts', 'split', 'training')
    validation_blks = cache_cfg.get_asset('artifacts', 'split', 'validation')
    domain_dict = cache_cfg.get_asset('artifacts', 'domain', 'by_block')

    # training blocks dirpath and artifact filepaths
    train_dir = f'./data/{dataset_name}/cache/training'
    # if inference blocks are present
    infer_dir = f'./data/{dataset_name}/cache/inference'
    _p = os.path.join(infer_dir, square_blks)
    square_blks_fpath = _p if os.path.exists(_p) else None

    # return the data summary dataclass
    return DataSummary(
        meta=_get_meta(
            dataset_name=dataset_name,
            fit_blocks_fpath=os.path.join(train_dir, train_val_blks),
            test_blocks_fpath=square_blks_fpath
        ),
        heads=_heads_stats(
            train_label_count_fpath=os.path.join(train_dir, lbl_count)
        ),
        data=_parse_datasets(
            train_blks_fpath=os.path.join(train_dir, training_blks),
            val_blks_fpath=os.path.join(train_dir, validation_blks),
            test_blks_fpath=square_blks_fpath
        ),
        doms=_get_domain(
            train_domain_fpath = os.path.join(train_dir, domain_dict),
            infer_domain_fpath = os.path.join(infer_dir, domain_dict)
        )
    )

# ------------------------------private  function------------------------------
def _get_meta(
    dataset_name: str,
    fit_blocks_fpath: str,
    test_blocks_fpath: str | None
) -> _Meta:
    '''Retrieve meta from a random valid block file.'''

    # declare types and default values
    fit_blk_b: int = 0
    test_blk_b: int = 0
    ignore_index: int = -1
    img_ch_num: int = 0

    # read a random block from valid blocks
    input_blks: dict[str, str] = utils.load_json(fit_blocks_fpath)
    rblk_fpath = random.choice(list(input_blks.values()))
    blk = dataset.DataBlock().load(rblk_fpath)
    blk.data.validate()
    # get array size
    img_size = math.prod(blk.data.image_normalized.shape) * 4 # float32 (4)
    lbl_size = math.prod(blk.data.label_masked.shape) * 8 # long (8)
    fit_blk_b = img_size + lbl_size # total
    # retrieve values for _Meta
    ignore_index = blk.meta['ignore_label']
    img_ch_num = blk.data.image_normalized.shape[0]

    if test_blocks_fpath is not None:
        # if inference blocks are present, read one as well
        input_blks: dict[str, str] = utils.load_json(test_blocks_fpath)
        rblk_fpath = random.choice(list(input_blks.values()))
        blk = dataset.DataBlock().load(rblk_fpath)
        # sanity: image channel should be the same for train and infer data
        if img_ch_num != 0: # meaning taken value from train_val blocks
            assert blk.data.image_normalized.shape[0] == img_ch_num
        else:
            img_ch_num = blk.data.image_normalized.shape[0]

        # get array size - image only
        test_blk_b = math.prod(blk.data.image_normalized.shape) * 4

    # check before return
    if fit_blk_b + test_blk_b == 0 or img_ch_num == 0:
        raise ValueError('Either fit or test data incomplete')
    return _Meta(dataset_name, fit_blk_b, test_blk_b, ignore_index, img_ch_num)

def _parse_datasets(
    train_blks_fpath: str,
    val_blks_fpath: str,
    test_blks_fpath: str | None
) -> _Data:
    '''Get data file lists.'''

    # training and validation datasets (must both be present)
    train_blks: dict[str, str] = utils.load_json(train_blks_fpath)
    val_blks: dict[str, str] = utils.load_json(val_blks_fpath)
    # inference blocks (optional)
    test_blks: dict[str, str] | None = None
    if test_blks_fpath is not None:
        test_blks = utils.load_json(test_blks_fpath)
    # return
    return _Data(train_blks, val_blks, test_blks)

def _heads_stats(train_label_count_fpath: str) -> _Heads:
    '''Get label class distribution and logits adjustments.'''

    # set up
    heads_w_counts: dict[str, list[int]] = {}
    logits_adjust: dict[str, list[float]] = {}
    topology: dict[str, dict[str, str | int | None]] = {}

    # iterate through label counts (training dataset)
    lbl_count: dict[str, list[int]] = utils.load_json(train_label_count_fpath)
    for layer_name, counts in lbl_count.items():
        if layer_name == 'original_label': # skip this for now
            continue

        # per-head label count and derived logit adjustment
        heads_w_counts[layer_name] = counts
        logits_adjust[layer_name] = _la_from_count(counts)

        # emit topology for current convention - from layer naming
        if layer_name == 'layer1':
            topology[layer_name] = {'head': None, 'cls': None}
        elif layer_name.startswith('layer2_'):
            cls_id = int(layer_name.split('layer2_')[1])
            topology[layer_name] = {'head': 'layer1', 'cls': cls_id}
        else:
            # if future names appear, one can decide to raise or set None
            topology[layer_name] = {'head': None, 'cls': None}

    # return the dicts
    return _Heads(heads_w_counts, logits_adjust, topology)

def _la_from_count(ct: list[int], t: float=1.0, e: float=1e-6) -> list[float]:
    '''Long-Tailed Recognition via Logit Adjustment Menon et al 2021.'''

    frequencies = [c / sum(ct) for c in ct]
    return [-t * math.log10(max(x, e)) for x in frequencies]

def _get_domain(
    train_domain_fpath: str,
    infer_domain_fpath: str
) -> _Domain:
    '''doc'''

    train_dom, train_meta = _domain(train_domain_fpath)
    infer_dom, infer_meta = _domain(infer_domain_fpath)
    return _Domain(train_dom, train_meta, infer_dom, infer_meta)

def _domain(domain_fpath: str) -> tuple[dict[str, dict] | None, dict | None]:
    '''Read domain data with type checks'''

    # index domain by block name if provided
    if os.path.exists(domain_fpath):
        domain: dict[str, dict[str, int | list[float]]] = {}
        meta: dict[str, dict[str, str | int | list]] = {}
        domain_src = utils.load_json(domain_fpath)
        assert isinstance(domain_src, list)
        # iterate through the original loaded json
        for dom in domain_src:
            assert isinstance(dom, dict)
            blkid = dom.get('block_name')
            assert isinstance(blkid, str)
            domain[blkid] = {k: v for k, v in dom.items() if k != 'block_name'}
        # get a sample
        dom_sample = domain[next(iter(domain))]
        dom_name_type = _dom_type(dom_sample)
        # assess each domain by its type
        for dom_name, dom_type in dom_name_type.items():
            if dom_type == 'ids':
                # iterate through the domain to get id range
                ids = set()
                for dom in domain.values():
                    v = dom[dom_name]
                    assert isinstance(v, int) # sanity
                    ids.add(v)
                meta[dom_name] = {
                    'type': 'ids',
                    'range': [min(ids), max(ids)]
                }
            if dom_type == 'vec':
                # ierate through the domain to affirm the vec dims
                dim = set()
                for dom in domain.values():
                    v = dom[dom_name]
                    assert isinstance(v, list) # sanity
                    dim.add(len(v))
                assert len(dim) == 1
                meta[dom_name] = {
                    'type': 'vec',
                    'dims': next(iter(dim))
                }

        return domain, meta
    # otherwise return None
    return None, None

def _dom_type(dom_sample: dict[str, int | list[float]]) -> dict[str, str]:
    meta: dict[str, str] | None = None
    if dom_sample is not None:
        meta = {}
        for k, v in dom_sample.items():
            if isinstance(v, int):
                meta[k] = 'ids'
            elif isinstance(v, list):
                meta[k] = 'vec'
            else:
                raise ValueError(f'Unrecognized domain type: {k}: {v}')
    return meta
