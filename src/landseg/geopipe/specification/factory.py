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
import landseg.core as core
import landseg.geopipe.core as geocore
import landseg.geopipe.utils as geo_utils
import landseg.utils as utils

def build_dataspec(
    catalog_meta_fpath: str,
    ids_domain_fpath: str | None,
    vec_domain_fpath: str | None,
    transform_schema_fpath: str,
    *,
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

    # load artifacts
    # catalog meta
    meta: geocore.CatalogMeta = utils.load_json(catalog_meta_fpath)
    # domains
    if ids_domain_fpath:
        payload = utils.load_json(ids_domain_fpath)
        ids_domain = geocore.DomainTileMap.from_payload(payload)
    else:
        ids_domain = None
    if vec_domain_fpath:
        payload = utils.load_json(vec_domain_fpath)
        vec_domain = geocore.DomainTileMap.from_payload(payload)
    else:
        vec_domain = None
    # transform schema
    schema: geocore.TransformSchema = utils.load_json(transform_schema_fpath)

    # return specs
    specs = core.DataSpecs(
        name=meta['dataset']['name'],
        mode='default',
        meta=_get_meta(meta, schema),
        heads=_get_heads(meta, schema),
        splits=_get_split(schema),
        domains=_get_domain(schema, ids_domain, vec_domain)
    )

    # print to screen and return
    if print_out:
        print(specs)
    return specs

# ------------------------------private  function------------------------------
def _get_meta(
    meta: geocore.CatalogMeta,
    schema: geocore.TransformSchema
) -> core.Meta:
    '''Populate `_Meta` dataclass from schema dictionary.'''

    # expected tensor sizes
    # per-pixel byte size
    img_b = numpy.dtype(meta['io_conventions']['dtypes']['image']).itemsize
    lbl_b = numpy.dtype(meta['io_conventions']['dtypes']['label']).itemsize
    # total pixels per tensor
    img_px = math.prod(meta['tensor_shapes']['image']['shape'])
    lbl_px = math.prod(meta['tensor_shapes']['label']['shape'])

    # return
    return core.Meta(
        img_ch=meta['tensor_shapes']['image']['C'],
        img_h_w=meta['tensor_shapes']['image']['H'],
        ignore_index=meta['io_conventions']['ignore_index'],
        img_arr_key=schema['image_array_key'],
        lbl_arr_key=schema['label_array_key'],
        blk_bytes=img_b * img_px + lbl_b * lbl_px,
    )

def _get_heads(
    meta: geocore.CatalogMeta,
    schema: geocore.TransformSchema
) -> core.Heads:
    '''Populate `_Heads` dataclass from schema dictionary.'''

    raw_counts: dict[str, list[int]] = schema['label_stats']
    counts = {k: v for k, v in raw_counts.items() if k != 'original'}
    return core.Heads(
        class_counts=counts,
        logits_adjust={k: __la_from_count(v) for k, v in counts.items()},
        head_parent=meta['labels']['channel_parent'],
        head_parent_cls=meta['labels']['channel_parent_cls']
    )

def __la_from_count(ct: list[int], t: float=1.0, e: float=1e-6) -> list[float]:
    '''Long-Tailed Recognition via Logit Adjustment Menon et al 2021.'''

    if sum(ct) == 0:
        return [0] * len(ct)
    frequencies = [c / sum(ct) for c in ct]
    return [-t * math.log10(max(x, e)) for x in frequencies]

def _get_split(schema: geocore.TransformSchema) -> core.Splits:
    '''Populate `_Split` dataclass from schema dictionary.'''

    return core.Splits(
        train=schema['train_blocks'],
        val=schema['val_blocks'],
        test=schema['test_blocks']
    )

def _get_domain(
    schema: geocore.TransformSchema,
    ids_domain: geocore.DomainTileMap | None,
    vec_domain: geocore.DomainTileMap | None
) -> core.Domains:
    '''Populate `_Domain` dataclass from schema dictionary.'''

    # get file paths
    train_blocks=schema['train_blocks']
    val_blocks=schema['val_blocks']
    test_blocks=schema['test_blocks']

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
    ids_domain: geocore.DomainTileMap | None,
    vec_domain: geocore.DomainTileMap | None
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
