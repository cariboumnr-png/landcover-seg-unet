'''Summarize the existing cache blocks and generate metadata.'''

# standard imports
from __future__ import annotations
import dataclasses
import math
import os
import random
# local imports
import _types
import dataset
import utils

# ------------------------------Public  Dataclass------------------------------
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

# ------------------------------private dataclass------------------------------
@dataclasses.dataclass
class _Meta:
    '''Metadata.'''
    dataset_name: str
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
    train: dict[str, str]
    val: dict[str, str]
    infer: dict[str, str] | None

@dataclasses.dataclass
class _Domain:
    '''Domain related.'''
    data: dict[str, dict[str, int | list[float]]] | None
    meta: dict[str, dict[str, str | int | list]] | None

# -------------------------------Public Function-------------------------------
def generate_summary(
        dataset_name: str,
        cache_config: _types.ConfigType
    ) -> DataSummary:
    '''Wrapper function to generate concrete `DataSummary` class.'''

    # config accessor
    cache_cfg = utils.ConfigAccess(cache_config)

    # get artifacts names
    lbl_count = cache_cfg.get_asset('artifacts', 'label', 'count_training')
    square_blks = cache_cfg.get_asset('artifacts', 'blocks', 'square')
    valid_blks = cache_cfg.get_asset('artifacts', 'blocks', 'valid')
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
        meta=_read_rand_block_meta(
            dataset_name=dataset_name,
            valid_blks_fpath=os.path.join(train_dir, valid_blks),
            infer_blks_fpath=square_blks_fpath
        ),
        heads=_heads_stats(
            label_count_fpath=os.path.join(train_dir, lbl_count)
        ),
        data=_read_datasets(
            train_blks_fpath=os.path.join(train_dir, training_blks),
            val_blks_fpath=os.path.join(train_dir, validation_blks),
            infer_blks_fpath=square_blks_fpath
        ),
        dom=_get_domain(
            domain_fpath = os.path.join(train_dir, domain_dict)
        )
    )

# ------------------------------private  function------------------------------
def _read_rand_block_meta(
        dataset_name: str,
        valid_blks_fpath: str,
        infer_blks_fpath: str | None = None
    ) -> _Meta:
    '''Retrieve meta from a random valid block file.'''

    # read a random block from valid blocks
    valid_blks: dict[str, str] = utils.load_json(valid_blks_fpath)
    rblk_fpath = random.choice(list(valid_blks.values()))
    blk = dataset.DataBlock().load(rblk_fpath)
    blk.data.validate()
    # retrieve meta
    ignore_index = blk.meta['ignore_label']
    img_ch_num=blk.data.image_normalized.shape[0]

    # if inference blocks are present, read one as well
    if infer_blks_fpath is not None:
        infer_blks: dict[str, str] = utils.load_json(infer_blks_fpath)
        rblk_fpath = random.choice(list(infer_blks.values()))
        blk = dataset.DataBlock().load(rblk_fpath)
        # sanity: image channel should be the same for train and infer data
        assert blk.data.image_normalized.shape[0] == img_ch_num

    # return
    return _Meta(dataset_name, ignore_index, img_ch_num)

def _read_datasets(
        train_blks_fpath: str,
        val_blks_fpath: str,
        infer_blks_fpath: str | None
    ) -> _Data:
    '''Get data file lists.'''

    # training and validation datasets (must both be present)
    training_blks: dict[str, str] = utils.load_json(train_blks_fpath)
    validation_blks: dict[str, str] = utils.load_json(val_blks_fpath)
    # inference blocks (optional)
    inference_blks: dict[str, str] | None = None
    if infer_blks_fpath is not None:
        inference_blks = utils.load_json(infer_blks_fpath)
    # return
    return _Data(training_blks, validation_blks, inference_blks)

def _heads_stats(label_count_fpath: str) -> _Heads:
    '''Get label class distribution and logits adjustments.'''

    # set up
    heads_w_counts: dict[str, list[int]] = {}
    logits_adjust: dict[str, list[float]] = {}
    topology: dict[str, dict[str, str | int | None]] = {}

    # iterate through label counts (training dataset)
    lbl_count = utils.load_json(label_count_fpath)
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

def _get_domain(domain_fpath: str) -> _Domain:
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

        return _Domain(domain, meta)
    # otherwise return None
    return _Domain(None, None)

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
