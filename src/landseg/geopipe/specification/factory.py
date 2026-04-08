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
Load dataset catalog metadata from the foundation stage and the finalized
schema from the transform stage to construct runtime data specifications.

This module bridges persisted dataset artifacts and the training stack
by assembling a `DataSpecs` object consumed by models and trainers.
'''

# standard imports
import math
# third-party imports
import numpy
# local imports
import landseg.artifacts as artifacts
import landseg.core as core
import landseg.geopipe.core as geo_core
import landseg.geopipe.utils as geo_utils

def build_dataspec(
    artifact_paths: artifacts.ArtifactPaths,
    *,
    ids_domain_name: str | None = None,
    vec_domain_name: str | None = None,
    print_out: bool = False
):
    '''
    Build a `DataSpecs` object from catalog metadata and transform schema.

    Loads:
    - Dataset-level catalog metadata from the foundation stage
    - Optional categorical and vectorized domain tile maps
    - The finalized transform schema that defines splits, heads, and
    artifacts

    These components are combined into a unified `DataSpecs` instance
    used by loaders, losses, metrics, and model runners.

    Args:
        catalog_meta_fpath: Path to dataset-level catalog metadata JSON.
        ids_domain_fpath: Optional path to categorical domain map JSON.
        vec_domain_fpath: Optional path to vector domain map JSON.
        transform_schema_fpath: Path to the transform schema JSON.
        print_out: If True, print the constructed `DataSpecs` to stdout.

    Returns:
        DataSpecs instance describing dataset structure and artifacts.
    '''

    # artifact fpaths
    data_schema_fpath = artifact_paths.foundation.data_blocks.dev.schema
    transform_schema_fpath = artifact_paths.transform.schema

    # load artifacts
    # domains
    _paths = artifact_paths.foundation.domains
    if ids_domain_name:
        ids_domain = _load_domain(_paths.domain_map_fpath(ids_domain_name))
    else:
        ids_domain = None
    if vec_domain_name:
        vec_domain = _load_domain(_paths.domain_map_fpath(vec_domain_name))
    else:
        vec_domain = None

    # data schema
    data_ctrl = artifacts.Controller[geo_core.DataSchema].load_json_or_fail
    data_schema = data_ctrl(data_schema_fpath).fetch()
    assert data_schema # typing assertion

    # transform schema
    transform_ctrl = artifacts.Controller[geo_core.TransformSchema].load_json_or_fail
    transform_schema = transform_ctrl(transform_schema_fpath).fetch()
    assert transform_schema # typing assertion

    # return specs
    specs = core.DataSpecs(
        name=data_schema['dataset']['name'],
        mode='default',
        meta=_get_meta(data_schema, transform_schema),
        heads=_get_heads(data_schema, transform_schema),
        splits=_get_split(transform_schema),
        domains=_get_domain(transform_schema, ids_domain, vec_domain)
    )

    # print to screen and return
    if print_out:
        print(specs)
    return specs

# ------------------------------private  function------------------------------
def _load_domain(fp: str) -> geo_core.DomainTileMap | None:
    '''doc'''

    # load payload and meta json
    D = dict[str, geo_core.DomainTile]
    M = geo_core.DomainMeta
    DomainCtrl = artifacts.PayloadController[D, M]
    schema = geo_core.DomainTileMap.SCHEMA_ID
    ctrl = DomainCtrl(fp, schema, artifacts.LifecyclePolicy.LOAD_OR_FAIL)
    payload = ctrl.load()
    assert payload # typing assertion
    return geo_core.DomainTileMap.from_json_payload(payload)

def _get_meta(
    data_schema: geo_core.DataSchema,
    transform_schema: geo_core.TransformSchema
) -> core.Meta:
    '''Populate `_Meta` dataclass from schema dictionary.'''

    # expected tensor sizes
    # per-pixel byte size
    img_b = numpy.dtype(data_schema['io_conventions']['dtypes']['image']).itemsize
    lbl_b = numpy.dtype(data_schema['io_conventions']['dtypes']['label']).itemsize
    # total pixels per tensor
    img_px = math.prod(data_schema['tensor_shapes']['image']['shape'])
    lbl_px = math.prod(data_schema['tensor_shapes']['label']['shape'])

    # get the patch grid of the test blocks if provided
    col, row = 0, 0
    # get block names sorted by col then row
    sorted_blknames = sorted(transform_schema['test_blocks'].keys(), key=geo_utils.name_xy)
    # get xy origin
    xmin, ymin = geo_utils.name_xy(sorted_blknames[0])
    # track max col and row number (0-based)
    for blkname in sorted_blknames:
        x, y = geo_utils.name_xy(blkname)
        col = max(col, (x - xmin) / data_schema['tensor_shapes']['image']['W'])
        row = max(row, (y - ymin) / data_schema['tensor_shapes']['image']['H'])

    # return
    return core.Meta(
        img_ch=data_schema['tensor_shapes']['image']['C'],
        img_h_w=data_schema['tensor_shapes']['image']['H'],
        ignore_index=data_schema['io_conventions']['ignore_index'],
        img_arr_key=transform_schema['image_array_key'],
        lbl_arr_key=transform_schema['label_array_key'],
        blk_bytes=img_b * img_px + lbl_b * lbl_px,
        test_blks_grid=(int(col + 1), int(row + 1))
    )

def _get_heads(
    data_schema: geo_core.DataSchema,
    transform_schema: geo_core.TransformSchema
) -> core.Heads:
    '''Populate `_Heads` dataclass from schema dictionary.'''

    raw_counts: dict[str, list[int]] = transform_schema['label_stats']
    counts = {k: v for k, v in raw_counts.items() if k != 'original'}
    return core.Heads(
        class_counts=counts,
        logits_adjust={k: __la_from_count(v) for k, v in counts.items()},
        head_parent=data_schema['labels']['channel_parent'],
        head_parent_cls=data_schema['labels']['channel_parent_cls']
    )

def __la_from_count(ct: list[int], t: float=1.0, e: float=1e-6) -> list[float]:
    '''Long-Tailed Recognition via Logit Adjustment Menon et al 2021.'''

    if sum(ct) == 0:
        return [0] * len(ct)
    frequencies = [c / sum(ct) for c in ct]
    return [-t * math.log10(max(x, e)) for x in frequencies]

def _get_split(transform_schema: geo_core.TransformSchema) -> core.Splits:
    '''Populate `_Split` dataclass from schema dictionary.'''

    return core.Splits(
        train=transform_schema['train_blocks'],
        val=transform_schema['val_blocks'],
        test=transform_schema['test_blocks']
    )

def _get_domain(
    transform_schema: geo_core.TransformSchema,
    ids_domain: geo_core.DomainTileMap | None,
    vec_domain: geo_core.DomainTileMap | None
) -> core.Domains:
    '''Populate `_Domain` dataclass from schema dictionary.'''

    # get file paths
    train_blocks=transform_schema['train_blocks']
    val_blocks=transform_schema['val_blocks']
    test_blocks=transform_schema['test_blocks']

    # format domains
    train_domain = __parse_domain(train_blocks, ids_domain, vec_domain)
    val_domain = __parse_domain(val_blocks, ids_domain, vec_domain)
    test_domain = __parse_domain(test_blocks, ids_domain, vec_domain)

    return core.Domains(
        train=train_domain,
        val=val_domain,
        test=test_domain,
        ids_max=ids_domain.max_id if ids_domain else -1,
        vec_dim=vec_domain.n_pca_ax if vec_domain else 0
    )

def __parse_domain(
    input_blocks: dict[str, str],
    ids_domain: geo_core.DomainTileMap | None,
    vec_domain: geo_core.DomainTileMap | None
) -> core.Domains.Dom:
    '''Parse blocks into discrete and vector domain mappings.'''

    # early exit
    if not input_blocks:
        return {'ids_domain': None, 'vec_domain': None}

    # prep
    output_ids_domain: dict[str, int] = {}
    output_vec_domain: dict[str, list[float]] = {}

    # index domain if provided
    if ids_domain:
        for coord, dom in ids_domain.items():
            blkname = geo_utils.xy_name(coord)
            if blkname in input_blocks:
                if dom['majority'] is None or dom['majority'] < 0:
                    output_ids_domain[blkname] = 0 # default value
                else:
                    output_ids_domain[blkname] = dom['majority']

    # vector domain if provided
    if vec_domain:
        for coord, dom in vec_domain.items():
            blkname = geo_utils.xy_name(coord)
            if blkname in input_blocks:
                if dom['pca_feature'] is None:
                    output_vec_domain[blkname] = [0.0] * vec_domain.n_pca_ax
                else:
                    output_vec_domain[blkname] = dom['pca_feature']

    # return
    return {'ids_domain': output_ids_domain, 'vec_domain': output_vec_domain}
