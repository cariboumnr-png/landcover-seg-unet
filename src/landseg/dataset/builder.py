'''Summarize the existing cache blocks and generate metadata.'''

# standard imports
from __future__ import annotations
import dataclasses
import math
import typing
# third-party imports
import numpy
# local imports
import landseg.domain as domain
import landseg.utils as utils

# ------------------------------Public  Dataclass------------------------------
@dataclasses.dataclass
class DataSpecs:
    '''doc'''
    meta: _Meta
    heads: _Heads
    splits: _Splits
    domains: _Domains

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
    test_blks_grid: tuple[int, int]

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
class _Domains:
    '''Domain related.'''
    train: _Dom
    val: _Dom
    test: _Dom
    ids_max: int
    vec_dim: int

    class _Dom(typing.TypedDict):
        '''doc'''
        ids_domain: dict[str, int] | None
        vec_domain: dict[str, list[float]] | None

# -------------------------------Public Function-------------------------------
def build_dataspec(
    schema_fpath: str,
    ids_domain: domain.DomainTileMap | None,
    vec_domain: domain.DomainTileMap | None
) -> DataSpecs:
    '''doc'''

    # read schema
    schema: dict[str, typing.Any] = utils.load_json(schema_fpath)

    return DataSpecs(
        meta=_get_meta(schema),
        heads=_get_heads(schema),
        splits=_get_split(schema),
        domains=_get_domain(schema, ids_domain, vec_domain)
    )

# ------------------------------private  function------------------------------
def _get_meta(schema: dict[str, typing.Any]) -> _Meta:
    '''doc'''

    # expected tensor sizes
    # per-pixel byte size
    img_b = numpy.dtype(schema['io_conventions']['dtypes']['image']).itemsize
    lbl_b = numpy.dtype(schema['io_conventions']['dtypes']['label']).itemsize
    # total pixels per tensor
    img_px = math.prod(schema['tensor_shapes']['image']['shape'])
    lbl_px = math.prod(schema['tensor_shapes']['label']['shape'])

    # get the patch grid of test blocks if provided
    col, row = 0, 0
    if schema['dataset']['has_test_data']:
        # get block names sorted by col then row
        blks: dict[str, str]
        blks = utils.load_json(schema['splits']['test_blocks']['fpath'])
        sorted_blknames = sorted(blks.keys(), key=__name_to_xy)
        # get xy origin
        xmin, ymin = __name_to_xy(sorted_blknames[0])
        # track max col and row number (0-based)
        for blkname in sorted_blknames:
            x, y = __name_to_xy(blkname)
            col = max(col, (x - xmin) / schema['world_grid']['tile_step_x'])
            row = max(row, (y - ymin) / schema['world_grid']['tile_step_y'])

    # return
    return _Meta(
        dataset_name=schema['dataset']['name'],
        fit_perblk_bytes=img_b * img_px,
        test_perblk_bytes=lbl_b * lbl_px,
        ignore_index=schema['io_conventions']['ignore_index'],
        img_ch_num=schema['tensor_shapes']['image']['C'],
        test_blks_grid=(int(col + 1), int(row + 1))
    )

def __name_to_xy(key: str) -> tuple[int, int]:
    '''doc'''

    # Expected pattern: 'col_<col>_row_<row>'
    _, col, _, row = key.split('_')
    return int(col), int(row)

def _get_heads(schema: dict[str, typing.Any]) -> _Heads:
    '''doc'''

    cc: dict[str, list[int]]
    cc = utils.load_json(schema['training_stats']['class_counts_train']['fpath'])
    return _Heads(
        class_counts={k: v for k, v in cc.items() if k != 'original_label'},
        logits_adjust={k: __la_from_count(v) for k, v in cc.items() if k !='original_label'},
        topology=schema['labels']['heads_topology']
    )

def __la_from_count(ct: list[int], t: float=1.0, e: float=1e-6) -> list[float]:
    '''Long-Tailed Recognition via Logit Adjustment Menon et al 2021.'''

    if sum(ct) == 0:
        return [0] * len(ct)
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
) -> _Domains:
    '''doc'''

    # get file paths
    train_fpath=schema['splits']['train_blocks']['fpath']
    val_fpath=schema['splits']['val_blocks']['fpath']
    test_fpath=schema['splits']['test_blocks'].get('fpath', None)

    # max id and number of pca axes
    max_ids = ids_domain.max_id if ids_domain else 0
    vec_dim = vec_domain.n_pca_ax if vec_domain else 0

    # format domains
    train_domain = __parse_domain(train_fpath, ids_domain, vec_domain)
    val_domain = __parse_domain(val_fpath, ids_domain, vec_domain)
    test_domain = __parse_domain(test_fpath, ids_domain, vec_domain)

    return _Domains(train_domain, val_domain, test_domain, max_ids, vec_dim)

def __parse_domain(
    blocks_fpath: str | None,
    ids_domain: domain.DomainTileMap | None,
    vec_domain: domain.DomainTileMap | None
) -> _Domains._Dom:
    '''doc'''

    # early exit
    if not blocks_fpath:
        return {'ids_domain': None, 'vec_domain': None}

    # prep
    input_blocks: dict[str, str] = utils.load_json(blocks_fpath)
    output_ids_domain: dict[str, int] = {}
    output_vec_domain: dict[str, list[float]] = {}

    # index domain if provided
    if ids_domain:
        for (x, y), dom in ids_domain.items():
            blkname = f'col_{x:06d}_row_{y:06d}'
            if blkname in input_blocks:
                if dom['majority'] is None:
                    output_ids_domain[blkname] = 0 # default value
                else:
                    output_ids_domain[blkname] = dom['majority']

    # vector domain if provided
    if vec_domain:
        for (x, y), dom in vec_domain.items():
            blkname = f'col_{x:06d}_row_{y:06d}'
            if blkname in input_blocks:
                if dom['pca_feature'] is None:
                    output_vec_domain[blkname] = [1.0] * vec_domain.n_pca_ax
                else:
                    output_vec_domain[blkname] = dom['pca_feature']

    # return
    return {'ids_domain': output_ids_domain, 'vec_domain': output_vec_domain}
