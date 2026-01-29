'''Summarize the existing cache blocks and generate metadata.'''

from __future__ import annotations
# standard imports
import dataclasses
import math
import os
import random
# local imports
import _types
import dataset
import utils

@dataclasses.dataclass
class DataSummary:
    '''doc'''
    meta: _Meta
    heads: _Heads
    data: _Data
    dom: _Domain

    def __post_init__(self):
        assert self.heads.class_counts.keys() == self.heads.logits_adjust.keys()
        if self.dom.data is not None:
            assert len(self.data.train) + len(self.data.val) == len(self.dom.data)

    def __repr__(self) -> str:
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
            f'Ignore index: {self.meta.ignore_index}',
            f'Number of image channels: {self.meta.img_ch_num}',
            f'Class distribution of each head: \n - {t1}',
            f'Head-wise logits adjustment: \n - {t2}',
            f'Number of training blocks: {len(self.data.train)}',
            f'Number of validation blocks: {len(self.data.val)}',
            f'Domain knowledge includes: {self.dom.meta}'
        ])

@dataclasses.dataclass
class _Meta:
    '''Metadata.'''
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
    train: list[str]
    val: list[str]

@dataclasses.dataclass
class _Domain:
    '''Domain related.'''
    data: dict[str, dict[str, int | list[float]]] | None
    meta: dict[str, dict[str, str | int | list]] | None

def generate(
        dataset_name: str,
        cache_config: _types.ConfigType
    ) -> DataSummary:
    '''Wrapper function to generate concrete `DataSummary` class.'''

    # config accessor
    cache_cfg = utils.ConfigAccess(cache_config)

    # get artifacts names
    lbl_count = cache_cfg.get_asset('artifacts', 'label', 'count_training')
    valid_blks = cache_cfg.get_asset('artifacts', 'blocks', 'valid')
    training_blks = cache_cfg.get_asset('artifacts', 'split', 'training')
    validation_blks = cache_cfg.get_asset('artifacts', 'split', 'validation')
    domain_dict = cache_cfg.get_asset('artifacts', 'domain', 'by_block')

    # training blocks dirpath and artifact filepaths
    _dir = f'./data/{dataset_name}/cache/training'
    train_lblstats_fpath = os.path.join(_dir, lbl_count)
    blks_fpath = os.path.join(_dir, valid_blks)
    train_datablks_fpaths = os.path.join(_dir, training_blks)
    val_datablks_fpaths = os.path.join(_dir, validation_blks)
    domain_fpath = os.path.join(_dir, domain_dict)

    # return the data summary dataclass
    return DataSummary(
        meta=_read_rand_block_meta(blks_fpath),
        heads=_heads_stats(train_lblstats_fpath),
        data=_read_datasets(train_datablks_fpaths, val_datablks_fpaths),
        dom=_get_domain(domain_fpath)
    )

def _read_rand_block_meta(validblks_fpath: str) -> _Meta:
    '''Retrieve meta from a random valid block file.'''

    # read a random block from valid blocks
    valid_blks: dict[str, str] = utils.load_json(validblks_fpath)
    rblk_fpath = random.choice(list(valid_blks.values()))
    blk = dataset.DataBlock().load(rblk_fpath)
    blk.data.validate()
    # return
    return _Meta(blk.meta['ignore_label'], blk.data.image_normalized.shape[0])

def _read_datasets(train_fpaths: str, val_fpaths: str) -> _Data:
    '''Get data file lists.'''

    # training and validation datasets
    training_blks: dict[str, str] = utils.load_json(train_fpaths)
    validation_blks: dict[str, str] = utils.load_json(val_fpaths)

    training_blks_fpaths = list(training_blks.values())
    validation_blk_fpaths = list(validation_blks.values())

    return _Data(training_blks_fpaths, validation_blk_fpaths)

def _heads_stats(fp: str) -> _Heads:
    '''Get label class distribution and logits adjustments.'''

    # set up
    heads_w_counts: dict[str, list[int]] = {}
    logits_adjust: dict[str, list[float]] = {}
    topology: dict[str, dict[str, str | int | None]] = {}

    # iterate through label counts (training dataset)
    lbl_count = utils.load_json(fp)
    for layer_name, counts in lbl_count.items():
        if layer_name == 'original_label': # skip this for now
            continue

        # per-head label count and derived logit adjustment
        heads_w_counts[layer_name] = counts
        logits_adjust[layer_name] = _la_from_count(counts)

        # Emit topology for current convention - from layer naming
        if layer_name == 'layer1':
            topology[layer_name] = {'head': None, 'cls': None}
        elif layer_name.startswith('layer2_'):
            cls_id = int(layer_name.split('layer2_')[1])
            topology[layer_name] = {'head': 'layer1', 'cls': cls_id}
        else:
            # If future names appear, you can decide to raise or set None
            topology[layer_name] = {'head': None, 'cls': None}

    # return the dicts
    return _Heads(heads_w_counts, logits_adjust, topology)

def _la_from_count(ct: list[int], t: float=1.0, e: float=1e-6) -> list[float]:
    '''Long-Tailed Recognition via Logit Adjustment Menon et al 2021.'''

    frequencies = [c / sum(ct) for c in ct]
    return [-t * math.log10(max(x, e)) for x in frequencies]

def _get_domain(fp: str) -> _Domain:
    '''Read domain data with type checks'''

    # index domain by block name if provided
    if os.path.exists(fp):
        domain: dict[str, dict[str, int | list[float]]] = {}
        meta: dict[str, dict[str, str | int | list]] = {}
        domain_src = utils.load_json(fp)
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
            if dom_type == 'id':
                # iterate through the domain to get id range
                ids = set()
                for dom in domain.values():
                    v = dom[dom_name]
                    assert isinstance(v, int) # sanity
                    ids.add(v)
                meta[dom_name] = {
                    'type': 'id',
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

        return _Domain(domain, meta)
    # otherwise return None
    return _Domain(None, None)

def _dom_type(dom_sample: dict[str, int | list[float]]) -> dict[str, str]:
    meta: dict[str, str] | None = None
    if dom_sample is not None:
        meta = {}
        for k, v in dom_sample.items():
            if isinstance(v, int):
                meta[k] = 'id'
            elif isinstance(v, list):
                meta[k] = 'vec'
            else:
                raise ValueError(f'Unrecognized domain type: {k}: {v}')
    return meta
