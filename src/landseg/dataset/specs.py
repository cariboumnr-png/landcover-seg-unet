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
Summarize cached blocks and emit dataset specifications for training and
evaluation. Builds a consolidated DataSpecs object from a dataset schema
(or from a single block) including meta, heads, splits, and domain info.

Public APIs:
    - DataSpecs: Dataclass container describing dataset specs.
    - build_dataspec: Construct DataSpecs from a full dataset schema.
    - build_dataspec_from_a_block: Construct minimal DataSpecs from one
      block (overfit mode).
'''

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
    '''Container for dataset specs used by trainers and models'''
    meta: _Meta         # general dataset metadata
    heads: _Heads       # head-wise label statistics and topology
    splits: _Splits     # train/val/test block file mappings
    domains: _Domains   # discrete/continuous domain metadata for conditioning

    def __str__(self) -> str:
        return '\n'.join([
            'Dataset summary:\n----------------------------------------',
            str(self.meta),
            str(self.heads),
            str(self.splits),
            str(self.domains)
        ])

# ------------------------------private dataclass------------------------------
@dataclasses.dataclass
class _Meta:
    '''General dataset metadata.'''
    dataset_name: str
    img_ch_num: int
    ignore_index: int
    fit_perblk_bytes: int
    test_blks_grid: tuple[int, int]
    single_block_mode: bool

    def __str__(self) -> str:
        return '\n'.join([
            '[General Meta]',
            f'Dataset name: {self.dataset_name}',
            f'Number of image channels: {self.img_ch_num}',
            f'Ignore index: {self.ignore_index}'
        ])

@dataclasses.dataclass
class _Heads:
    '''Head-wise label statistics and topology.'''
    class_counts: dict[str, list[int]]
    logits_adjust: dict[str, list[float]]
    topology: dict[str, dict[str, str | int | None]]

    def __str__(self) -> str:
        def _ln(lst):
            return [round(x, 2) for x in lst]
        cc = self.class_counts
        la = self.logits_adjust
        t1 = '\n - '.join([f'{k}:\t{v}' for k, v in cc.items()])
        t2 = '\n - '.join([f'{k}:\t{_ln(v)}' for k, v in la.items()])
        return '\n'.join([
            '[Heads Specs]',
            f'Class distribution of each head: \n - {t1}',
            f'Head-wise logits adjustment: \n - {t2}',
        ])

@dataclasses.dataclass
class _Splits:
    '''Train/val/test block file mappings.'''
    train: dict[str, str]
    val: dict[str, str]
    test: dict[str, str] | None

    def __str__(self) -> str:
        return '\n'.join([
            '[Dataset Split]',
            f'Number of train blocks: {len(self.train)}',
            f'Number of val blocks: {len(self.val)}',
            f'Number of test blocks: {len(self.test or {})}',
        ])

@dataclasses.dataclass
class _Domains:
    '''Domain metadata for conditioning.'''
    train: _Dom
    val: _Dom
    test: _Dom
    ids_max: int
    vec_dim: int

    class _Dom(typing.TypedDict):
        '''Typed domain dictionaries.'''
        ids_domain: dict[str, int] | None
        vec_domain: dict[str, list[float]] | None

    def __str__(self) -> str:
        return '\n'.join([
            '[Domain Knowledge]',
            f'Discrete domain IDs count: {self.ids_max + 1}', # 0- to 1-based
            f'Continuous domain PCA number of axes: {self.vec_dim}'
        ])

# -------------------------------Public Function-------------------------------
def build_dataspec(
    schema_fpath: str,
    ids_domain: domain.DomainTileMap | None,
    vec_domain: domain.DomainTileMap | None
) -> DataSpecs:
    '''
    Build a `DataSpecs` instance from dataset schema and optional domains.

    Args:
        schema: Path to a schema JSON file.
        ids_domain: Optional domain map of discrete domain IDs by tile.
        vec_domain: Optional domain map of continuous vectors by tile.

    Returns:
        DataSpecs: Dataset specifications assembled from the schema and
            domain maps.

    Raises:
        FileNotFoundError: If a schema path is provided but not be found.
        ValueError: If the schema file is unreadable or malformed.
    '''

    # read schema
    schema_dict: dict[str, typing.Any] = utils.load_json(schema_fpath)
    # build return the class instance
    return DataSpecs(
        meta=_get_meta(schema_dict),
        heads=_get_heads(schema_dict),
        splits=_get_split(schema_dict),
        domains=_get_domain(schema_dict, ids_domain, vec_domain)
    )

def build_dataspec_from_a_block(schema: dict[str, typing.Any]) -> DataSpecs:
    '''
    Build a minimal DataSpecs instance from a single block schema.

    Args:
        schema: In-memory schema dictionary derived from one block.

    Returns:
        DataSpecs: Minimal specs suitable for overfit/sanity tests.
    '''

    # return directly from schema dict
    return DataSpecs(
        meta =_Meta(
            dataset_name=schema['dataset_name'],
            img_ch_num=schema['image_channel'],
            ignore_index=schema['ignore_index'],
            fit_perblk_bytes=0,
            test_blks_grid=(0, 0),
            single_block_mode=True
        ),
        heads=_Heads(
            class_counts=schema['class_counts'],
            logits_adjust=schema['logit_adjust'],
            topology=schema['topology']
        ),
        splits=_Splits(
            train=schema['train_split'],
            val=schema['val_split'],
            test=None
        ),
        domains=_Domains(
            train={'ids_domain': None, 'vec_domain': None},
            val={'ids_domain': None, 'vec_domain': None},
            test={'ids_domain': None, 'vec_domain': None},
            ids_max=-1,
            vec_dim=0
        )
    )

# ------------------------------private  function------------------------------
def _get_meta(schema: dict[str, typing.Any]) -> _Meta:
    '''Populate `_Meta` dataclass from schema dictionary.'''

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
        img_ch_num=schema['tensor_shapes']['image']['C'],
        ignore_index=schema['io_conventions']['ignore_index'],
        fit_perblk_bytes=img_b * img_px + lbl_b * lbl_px,
        test_blks_grid=(int(col + 1), int(row + 1)),
        single_block_mode=False
    )

def __name_to_xy(key: str) -> tuple[int, int]:
    '''Simple parser to get coordinates from block name.'''

    # Expected pattern: 'col_<col>_row_<row>'
    _, col, _, row = key.split('_')
    return int(col), int(row)

def _get_heads(schema: dict[str, typing.Any]) -> _Heads:
    '''Populate `_Heads` dataclass from schema dictionary.'''

    train_blks_path = schema['training_stats']['class_counts_train']['fpath']
    raw_counts: dict[str, list[int]] = utils.load_json(train_blks_path)
    counts = {k: v for k, v in raw_counts.items() if k != 'original_label'}
    return _Heads(
        class_counts=counts,
        logits_adjust={k: __la_from_count(v) for k, v in counts.items()},
        topology=schema['labels']['heads_topology']
    )

def __la_from_count(ct: list[int], t: float=1.0, e: float=1e-6) -> list[float]:
    '''Long-Tailed Recognition via Logit Adjustment Menon et al 2021.'''

    if sum(ct) == 0:
        return [0] * len(ct)
    frequencies = [c / sum(ct) for c in ct]
    return [-t * math.log10(max(x, e)) for x in frequencies]

def _get_split(schema: dict[str, typing.Any]) -> _Splits:
    '''Populate `_Split` dataclass from schema dictionary.'''

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
    '''Populate `_Domain` dataclass from schema dictionary.'''

    # get file paths
    train_fpath=schema['splits']['train_blocks']['fpath']
    val_fpath=schema['splits']['val_blocks']['fpath']
    test_fpath=schema['splits']['test_blocks'].get('fpath', None)

    # format domains
    train_domain = __parse_domain(train_fpath, ids_domain, vec_domain)
    val_domain = __parse_domain(val_fpath, ids_domain, vec_domain)
    test_domain = __parse_domain(test_fpath, ids_domain, vec_domain)

    return _Domains(
        train=train_domain,
        val=val_domain,
        test=test_domain,
        ids_max=ids_domain.max_id if ids_domain else -1,
        vec_dim=vec_domain.n_pca_ax if vec_domain else 0
    )

def __parse_domain(
    blocks_fpath: str | None,
    ids_domain: domain.DomainTileMap | None,
    vec_domain: domain.DomainTileMap | None
) -> _Domains._Dom:
    '''Parse blocks into discrete and vector domain mappings.'''

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
                if dom['majority'] is None or dom['majority'] < 0:
                    output_ids_domain[blkname] = 0 # default value
                else:
                    output_ids_domain[blkname] = dom['majority']

    # vector domain if provided
    if vec_domain:
        for (x, y), dom in vec_domain.items():
            blkname = f'col_{x:06d}_row_{y:06d}'
            if blkname in input_blocks:
                if dom['pca_feature'] is None:
                    output_vec_domain[blkname] = [0.0] * vec_domain.n_pca_ax
                else:
                    output_vec_domain[blkname] = dom['pca_feature']

    # return
    return {'ids_domain': output_ids_domain, 'vec_domain': output_vec_domain}
