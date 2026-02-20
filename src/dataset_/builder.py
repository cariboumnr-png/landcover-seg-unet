'''Summarize the existing cache blocks and generate metadata.'''

# standard imports
from __future__ import annotations
import copy
import dataclasses
import math
import typing
# third-party imports
import numpy
# local imports
import domain
import utils

# ------------------------------Public  Dataclass------------------------------
@dataclasses.dataclass
class DataSpec:
    '''doc'''
    meta: _Meta
    heads: _Heads
    splits: _Splits
    domains: _Domain

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
            f'Number of train blocks: {len(self.splits.train)}',
            f'Number of val blocks: {len(self.splits.val)}',
            f'Number of test blocks: {len(self.splits.val or {})}',
            f'Domain knowledge includes: {self.domains}'
        ])

# ------------------------------private dataclass------------------------------
@dataclasses.dataclass
class _Meta:
    '''Metadata.'''
    dataset_name: str
    fit_perblk_bytes: int
    test_perblk_bytes: int
    ignore_index: int
    img_ch_num: int

@dataclasses.dataclass
class _Heads:
    '''Heads label stats'''
    class_counts: dict[str, list[int]]
    logits_adjust: dict[str, list[float]]
    topology: dict[str, dict[str, str | int | None]]

@dataclasses.dataclass
class _Splits:
    '''Block file paths of training and validation datasets.'''
    train: dict[str, str]
    val: dict[str, str]
    test: dict[str, str] | None

@dataclasses.dataclass
class _Domain:
    '''Domain related.'''
    train_domain: dict[str, typing.Any]
    val_domain: dict[str, typing.Any]
    test_domain: dict[str, typing.Any] | None
    domain_max_ids: int
    domain_n_pca_ax: int

# -------------------------------Public Function-------------------------------
def build_dataspec(
    schema_fpath: str,
    ids_domain: domain.DomainTileMap | None,
    vec_domain: domain.DomainTileMap | None
) -> DataSpec:
    '''doc'''

    # read schema
    schema: dict[str, typing.Any] = utils.load_json(schema_fpath)

    return DataSpec(
        meta=_get_meta(schema),
        heads=_get_heads(schema),
        splits=_get_split(schema),
        domains=_get_domain(schema, ids_domain, vec_domain)
    )

# ------------------------------private  function------------------------------
def _get_meta(schema: dict[str, typing.Any]) -> _Meta:
    '''doc'''

    # per-pixel byte size
    img_b = numpy.dtype(schema['io_conventions']['dtypes']['image']).itemsize
    lbl_b = numpy.dtype(schema['io_conventions']['dtypes']['label']).itemsize
    # total pixels per tensor
    img_px = math.prod(schema['tensor_shapes']['image']['shape'])
    lbl_px = math.prod(schema['tensor_shapes']['label']['shape'])
    # return
    return _Meta(
        dataset_name=schema['dataset']['name'],
        fit_perblk_bytes=img_b * img_px,
        test_perblk_bytes=lbl_b * lbl_px,
        ignore_index=schema['io_conventions']['ignore_index'],
        img_ch_num=schema['tensor_shapes']['image']['C']
    )

def _get_heads(schema: dict[str, typing.Any]) -> _Heads:
    '''doc'''

    cc: dict[str, list[int]]
    cc = utils.load_json(schema['training_stats']['class_counts_train']['fpath'])
    return _Heads(
        class_counts={k: v for k, v in cc.items() if k != 'original_label'},
        logits_adjust={k: _la_from_count(v) for k, v in cc.items() if k !='original_label'},
        topology=schema['labels']['heads_topology']
    )

def _la_from_count(ct: list[int], t: float=1.0, e: float=1e-6) -> list[float]:
    '''Long-Tailed Recognition via Logit Adjustment Menon et al 2021.'''

    frequencies = [c / sum(ct) for c in ct]
    return [-t * math.log10(max(x, e)) for x in frequencies]

def _get_split(schema: dict[str, typing.Any]) -> _Splits:
    '''doc'''

    # get file paths
    train_fpath=schema['splits']['train_blocks']['fpath']
    val_fpath=schema['splits']['val_blocks']['fpath']
    test_fpath=schema['splits']['test_blocks'].get('fpath', None)

    return _Splits(
        train=utils.load_json(train_fpath),
        val=utils.load_json(val_fpath),
        test=utils.load_json(test_fpath) if test_fpath else None
    )

def _get_domain(
    schema: dict[str, typing.Any],
    ids_domain: domain.DomainTileMap | None,
    vec_domain: domain.DomainTileMap | None
) -> _Domain:
    '''doc'''

    # get file paths
    train_fpath=schema['splits']['train_blocks']['fpath']
    val_fpath=schema['splits']['val_blocks']['fpath']
    test_fpath=schema['splits']['test_blocks'].get('fpath', None)

    # max id and number of pca axes
    max_ids = ids_domain.max_id if ids_domain else 0
    vec_dim = vec_domain.n_pca_ax if vec_domain else 0

    # format domains
    train_domain = _parse_domain(train_fpath, ids_domain, vec_domain)
    val_domain = _parse_domain(val_fpath, ids_domain, vec_domain)
    if test_fpath:
        test_domain = _parse_domain(test_fpath, ids_domain, vec_domain)
    else:
        test_domain = None

    return _Domain(train_domain, val_domain, test_domain, max_ids, vec_dim)

def _parse_domain(
    blocks_fpath: str,
    ids_domain: domain.DomainTileMap | None,
    vec_domain: domain.DomainTileMap | None
) -> dict[str, domain.DomainTileMap | None]:
    '''doc'''

    # prep
    input_blocks: dict[str, str] = utils.load_json(blocks_fpath)
    blk_ids_domain = copy.deepcopy(ids_domain)
    blk_vec_domain = copy.deepcopy(vec_domain)

    # index domain if provided
    if blk_ids_domain:
        to_pop = []
        for coord in blk_ids_domain.keys():
            x, y = coord
            blkname = f'col_{x:06d}_row_{y:06d}'
            if blkname not in input_blocks:
                to_pop.append(coord)
        blk_ids_domain.pop_many(coords=to_pop)

    # vector domain if provided
    if blk_vec_domain:
        to_pop = []
        for coord in blk_vec_domain.keys():
            x, y = coord
            blkname = f'col_{x:06d}_row_{y:06d}'
            if blkname not in input_blocks:
                to_pop.append(coord)
        blk_vec_domain.pop_many(coords=to_pop)

    # return
    return {
        'ids_domain': blk_ids_domain,
        'vec_domain': blk_vec_domain
    }
