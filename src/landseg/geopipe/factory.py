# =========================================================================== #
#            Copyright © His Majesty the King in right of Ontario,            #
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
Construct runtime data specifications from ingestion and preparation schemas.

This module bridges persisted dataset artifacts and the training stack
by assembling a `DataSpecs` object consumed by models, trainers, and
evaluation routines.

Public APIs:
    - build_dataspec: Assembles runtime DataSpecs from artifacts.
'''

# standard imports
import math
import typing
# third-party imports
import numpy
import torch
# local imports
import landseg.artifacts as artifacts
import landseg.core as core
import landseg.geopipe.core as geo_core
import landseg.geopipe.utils as geo_utils
import landseg.knowledge as knowledge


# ----- public functions
def build_dataspec(
    artifact_paths: artifacts.ArtifactPaths,
    *,
    mode: typing.Literal['default', 'single', 'val_only', 'test_only'],
    ids_domain_name: str | None = None,
    vec_domain_name: str | None = None
) -> core.DataSpecs:
    '''
    Build runtime data specification from dataset artifacts and schemas.

    Assembles a `DataSpecs` object by loading dataset schema from
    ingestion, preparation schema defining splits and heads, and
    optional categorical or vectorized domain tilemaps.

    Args:
        artifact_paths:
            Hierarchical path manager for repository artifacts.
        mode:
            Execution mode governing dataset partition selection.
        ids_domain_name:
            Optional name of categorical domain tilemap artifact.
        vec_domain_name:
            Optional name of continuous/vector domain artifact.

    Returns:
        core.DataSpecs:
            Configured runtime specifications for dataset consumption.
    '''
    # artifact fpaths
    data_schema_fpath = artifact_paths.data_ingestion.data_blocks.dev.schema
    transform_schema_fpath = artifact_paths.data_preparation.schema

    # load artifacts
    # domains
    _paths = artifact_paths.data_ingestion.domains
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

    # transform schema
    transform_ctrl = (
        artifacts.Controller[geo_core.TransformSchema].load_json_or_fail
    )
    transform_schema = transform_ctrl(transform_schema_fpath).fetch()

    # return specs
    return core.DataSpecs(
        name=data_schema['dataset']['name'],
        mode=mode,
        meta=_get_meta(data_schema, transform_schema),
        heads=_get_heads(
            data_schema,
            transform_schema,
            knowledge_paths=artifact_paths.knowledge
        ),
        splits=_get_split(transform_schema),
        domains=_get_domain(transform_schema, ids_domain, vec_domain)
    )


# ----- private helpers
def _load_domain(fp: str) -> geo_core.DomainTileMap | None:
    '''Load a DomainTileMap from the specified JSON path.'''
    # load payload and meta json
    D = dict[str, geo_core.DomainTile]
    M = geo_core.DomainMeta
    DomainCtrl = artifacts.PayloadController[D, M]
    ctrl = DomainCtrl(
        fp,
        schema_id=geo_core.DomainTileMap.SCHEMA_ID,
        policy=artifacts.LifecyclePolicy.LOAD_OR_FAIL
    )
    payload = ctrl.load()
    assert payload # typing assertion
    return geo_core.DomainTileMap.from_json_payload(payload)


def _get_meta(
    data_schema: geo_core.DataSchema,
    transform_schema: geo_core.TransformSchema
) -> core.Meta:
    '''Populate core.Meta dataclass from schema dictionaries.'''
    # expected tensor sizes
    # per-pixel byte size
    img_b = numpy.dtype(
        data_schema['io_conventions']['dtypes']['image']
    ).itemsize
    lbl_b = numpy.dtype(
        data_schema['io_conventions']['dtypes']['label']
    ).itemsize
    # total pixels per tensor
    img_px = math.prod(data_schema['tensor_shapes']['image']['shape'])
    lbl_px = math.prod(data_schema['tensor_shapes']['label']['shape'])

    # get the patch grid of the test blocks if provided
    test_blocks = transform_schema.get('test_blocks', None)
    col, row = 0, 0
    if len(test_blocks) > 0:
        # get block names sorted by col then row
        sorted_blknames = sorted(test_blocks.keys(), key=geo_utils.name_xy)
        # get xy origin
        xmin, ymin = geo_utils.name_xy(sorted_blknames[0])
        # track max col and row number (0-based)
        img_w = data_schema['tensor_shapes']['image']['W']
        img_h = data_schema['tensor_shapes']['image']['H']
        for blkname in sorted_blknames:
            x, y = geo_utils.name_xy(blkname)
            col = max(col, (x - xmin) / img_w)
            row = max(row, (y - ymin) / img_h)
        col, row = int(col + 1), int(row + 1)
        # check if test blocks form a continuous array without gaps
        if col * row != len(sorted_blknames):
            col, row = 0, 0 # empty grid for downstream

    # return a meta dataclass
    return core.Meta(
        test_blks_grid=(col, row),
        blk_bytes=img_b * img_px + lbl_b * lbl_px,
        label_color_map=data_schema['labels']['label_color_map'],
        image_specs=core.Meta.Image(
            num_channels=data_schema['tensor_shapes']['image']['C'],
            height_width=data_schema['tensor_shapes']['image']['H'],
            array_key=transform_schema['image_array_key'],
            band_map=data_schema['io_conventions']['image_band_map'],
        ),
        label_specs=core.Meta.Label(
            array_key=transform_schema['label_array_key'],
            ignore_index=data_schema['io_conventions']['ignore_index']
        )
    )


def _get_heads(
    data_schema: geo_core.DataSchema,
    transform_schema: geo_core.TransformSchema,
    knowledge_paths: artifacts.KnowledgePaths | None = None
) -> core.Heads:
    '''Populate core.Heads dataclass from schema dictionary.'''
    raw_counts: dict[str, list[int]] = transform_schema['label_stats']
    counts = {k: v for k, v in raw_counts.items() if k != 'original'}
    taxonomy = data_schema['labels'].get('label_taxonomy', {})
    sim_matrices: dict[str, torch.Tensor] = {}
    kp = knowledge_paths or artifacts.KnowledgePaths()
    for hname, tax_dict in taxonomy.items():
        if isinstance(tax_dict, dict) and 'profile' in tax_dict:
            profile = tax_dict['profile']
            sim_matrices[hname] = knowledge.resolve_similarity_matrix(
                profile,
                knowledge_root=kp.root,
            )

    return core.Heads(
        class_counts=counts,
        logits_adjust={k: __la_from_count(v) for k, v in counts.items()},
        head_parent=data_schema['labels']['label_parent'],
        head_parent_cls=data_schema['labels']['label_parent_cls'],
        taxonomy=taxonomy,
        similarity_matrices=sim_matrices,
    )


def __la_from_count(
    ct: list[int],
    t: float = 1.0,
    e: float = 1e-6,
) -> list[float]:
    '''Long-tailed recognition via logit adjustment (Menon et al 2021).'''
    if sum(ct) == 0:
        return [0] * len(ct)
    frequencies = [c / sum(ct) for c in ct]
    return [-t * math.log10(max(x, e)) for x in frequencies]


def _get_split(transform_schema: geo_core.TransformSchema) -> core.Splits:
    '''Populate core.Splits dataclass from schema dictionary.'''
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
    '''Populate core.Domains dataclass from schema dictionary.'''
    # get file paths
    train_blocks = transform_schema['train_blocks']
    val_blocks = transform_schema['val_blocks']
    test_blocks = transform_schema['test_blocks']

    # format domains
    train_domain = __parse_domain(train_blocks, ids_domain, vec_domain)
    val_domain = __parse_domain(val_blocks, ids_domain, vec_domain)
    test_domain = __parse_domain(test_blocks, ids_domain, vec_domain)

    return core.Domains(
        train=train_domain,
        val=val_domain,
        test=test_domain,
        ids_num=ids_domain.max_id if ids_domain else -1,
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
