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
import math
import os
# third-party imports
import numpy
# local imports
import landseg.core.ingest_protocols as ingest_protocols
import landseg.utils as utils

# -------------------------------Public Function-------------------------------
def build_dataspec(
    schema_fpath: str,
    ids_domain: ingest_protocols.DomainTileMapLike | None,
    vec_domain: ingest_protocols.DomainTileMapLike | None
) -> ingest_protocols.DataSpecs:
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
    schema_dict: ingest_protocols.SchemaFull = utils.load_json(schema_fpath)
    # build return the class instance
    return ingest_protocols.DataSpecs(
        name=schema_dict['dataset']['name'],
        mode='default',
        meta=_get_meta(schema_dict),
        heads=_get_heads(schema_dict),
        splits=_get_split(schema_dict),
        domains=_get_domain(schema_dict, ids_domain, vec_domain)
    )

def build_dataspec_one_block(
        schema: ingest_protocols.SchemaOneBlock
    ) -> ingest_protocols.DataSpecs:
    '''
    Build a minimal DataSpecs instance from a single block schema.

    Args:
        schema: In-memory schema dictionary derived from one block.

    Returns:
        DataSpecs: Minimal specs suitable for overfit/sanity tests.
    '''

    # return directly from schema dict
    return ingest_protocols.DataSpecs(
        name=schema['dataset_name'],
        mode='single',
        meta =ingest_protocols.Meta(
            img_ch=schema['image_channel'],
            img_h_w=schema['image_h_w'],
            ignore_index=schema['ignore_index'],
            img_arr_key=schema['img_arr_key'],
            lbl_arr_key=schema['lbl_arr_key'],
            fit_perblk_bytes=0,
            test_blks_grid=(0, 0)
        ),
        heads=ingest_protocols.Heads(
            class_counts=schema['class_counts'],
            logits_adjust=schema['logit_adjust'],
            head_parent=schema['head_parent'],
            head_parent_cls=schema['head_parent_cls']
        ),
        splits=ingest_protocols.Splits(
            train=schema['train_split'],
            val=schema['val_split'],
            test=None
        ),
        domains=ingest_protocols.Domains(
            train={'ids_domain': None, 'vec_domain': None},
            val={'ids_domain': None, 'vec_domain': None},
            test={'ids_domain': None, 'vec_domain': None},
            ids_max=-1,
            vec_dim=0
        )
    )

# ------------------------------private  function------------------------------
def _get_meta(schema: ingest_protocols.SchemaFull) -> ingest_protocols.Meta:
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
        blks = utils.load_json(schema['splits']['test_blocks'])
        sorted_blknames = sorted(blks.keys(), key=__name_to_xy)
        # get xy origin
        xmin, ymin = __name_to_xy(sorted_blknames[0])
        # track max col and row number (0-based)
        for blkname in sorted_blknames:
            x, y = __name_to_xy(blkname)
            col = max(col, (x - xmin) / schema['world_grid']['tile_step_x'])
            row = max(row, (y - ymin) / schema['world_grid']['tile_step_y'])

    # return
    return ingest_protocols.Meta(
        img_ch=schema['tensor_shapes']['image']['C'],
        img_h_w=schema['tensor_shapes']['image']['H'],
        ignore_index=schema['io_conventions']['ignore_index'],
        img_arr_key=schema['io_conventions']['keys']['image_key'],
        lbl_arr_key=schema['io_conventions']['keys']['label_key'],
        fit_perblk_bytes=img_b * img_px + lbl_b * lbl_px,
        test_blks_grid=(int(col + 1), int(row + 1)),
    )

def __name_to_xy(key: str) -> tuple[int, int]:
    '''Simple parser to get coordinates from block name.'''

    # Expected pattern: 'col_<col>_row_<row>'
    _, col, _, row = key.split('_')
    return int(col), int(row)

def _get_heads(schema: ingest_protocols.SchemaFull) -> ingest_protocols.Heads:
    '''Populate `_Heads` dataclass from schema dictionary.'''

    train_blks_path = schema['training_stats']['class_counts_train']
    raw_counts: dict[str, list[int]] = utils.load_json(train_blks_path)
    counts = {k: v for k, v in raw_counts.items() if k != 'original_label'}
    return ingest_protocols.Heads(
        class_counts=counts,
        logits_adjust={k: __la_from_count(v) for k, v in counts.items()},
        head_parent=schema['labels']['head_parent'],
        head_parent_cls=schema['labels']['head_parent_cls']
    )

def __la_from_count(ct: list[int], t: float=1.0, e: float=1e-6) -> list[float]:
    '''Long-Tailed Recognition via Logit Adjustment Menon et al 2021.'''

    if sum(ct) == 0:
        return [0] * len(ct)
    frequencies = [c / sum(ct) for c in ct]
    return [-t * math.log10(max(x, e)) for x in frequencies]

def _get_split(schema: ingest_protocols.SchemaFull) -> ingest_protocols.Splits:
    '''Populate `_Split` dataclass from schema dictionary.'''

    # get file paths
    train_fpath=schema['splits']['train_blocks']
    val_fpath=schema['splits']['val_blocks']
    test_fpath=schema['splits']['test_blocks']

    return ingest_protocols.Splits(
        train=utils.load_json(train_fpath),
        val=utils.load_json(val_fpath),
        test=utils.load_json(test_fpath) if os.path.exists(test_fpath) else None
    )

def _get_domain(
    schema: ingest_protocols.SchemaFull,
    ids_domain: ingest_protocols.DomainTileMapLike | None,
    vec_domain: ingest_protocols.DomainTileMapLike | None
) -> ingest_protocols.Domains:
    '''Populate `_Domain` dataclass from schema dictionary.'''

    # get file paths
    train_fpath=schema['splits']['train_blocks']
    val_fpath=schema['splits']['val_blocks']
    test_fpath=schema['splits']['test_blocks']

    # format domains
    train_domain = __parse_domain(train_fpath, ids_domain, vec_domain)
    val_domain = __parse_domain(val_fpath, ids_domain, vec_domain)
    test_domain = __parse_domain(test_fpath, ids_domain, vec_domain)

    return ingest_protocols.Domains(
        train=train_domain,
        val=val_domain,
        test=test_domain,
        ids_max=ids_domain.max_id if ids_domain else -1,
        vec_dim=vec_domain.n_pca_ax if vec_domain else 0
    )

def __parse_domain(
    blocks_fpath: str,
    ids_domain: ingest_protocols.DomainTileMapLike | None,
    vec_domain: ingest_protocols.DomainTileMapLike | None
) -> ingest_protocols.Domains.Dom:
    '''Parse blocks into discrete and vector domain mappings.'''

    # early exit
    if not os.path.exists(blocks_fpath):
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
