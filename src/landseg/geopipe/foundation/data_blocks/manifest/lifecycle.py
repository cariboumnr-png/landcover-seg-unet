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
Data block catalog and schema artifact lifecycle management.

This module orchestrates creation, validation, and incremental updates of
dataset manifest artifacts, including `catalog.json` and `schema.json`.
It coordinates policy-driven rebuilds, detects newly created or modified
data blocks, and ensures consistency between on-disk block artifacts and
their recorded schema throughout data preparation and update workflows.
'''

# standard imports
import dataclasses
import os
import typing
# local imports
import landseg.artifacts as artifacts
import landseg.geopipe.core as geo_core
import landseg.geopipe.foundation.data_blocks.manifest as manifest
import landseg.geopipe.utils as geo_utils

# typing aliases
CatalogDictCtrl = artifacts.Controller[dict[str, geo_core.CatalogEntry]]
SchemaCtrl = artifacts.Controller[geo_core.DataSchema]

# ------------------------------Public  Dataclass------------------------------
@dataclasses.dataclass
class ManifestUpdateContext:
    '''Context describing a manifest update operation.'''
    updated_coords: list[tuple[int, int]]   # grid coords for created blocks
    source_image: str               # path to the source image raster
    source_label: str | None        # optional path to the label raster
    mapped_grid_id: str             # id for the grid the blocks are mapped to
    blocks_dir: str                 # where data blocks are
    label_color_map: dict[str, list[int]] | None

# -------------------------------Public Function-------------------------------
def update_manifest(
    context: ManifestUpdateContext,
    catalog_fpath: str,
    schema_fpath: str,
    *,
    policy: artifacts.LifecyclePolicy,
) -> dict[str, typing.Any]:
    '''
    Update dataset manifest artifacts according to lifecycle policy.

    This function manages coordinated updates to the dataset-level
    `catalog.json` and `schema.json` files. It determines the current
    state of cataloged block artifacts relative to blocks present on
    disk, applies the specified lifecycle policy to decide whether to
    rebuild or append entries, and writes updated JSON artifacts with
    integrity hashes.
    '''

    # ----- catalog
    ctrl = CatalogDictCtrl(catalog_fpath, policy)
    try:
        catalog_dict = ctrl.fetch()
    except artifacts.ArtifactError as exc: # e.g., current catalog corrupted
        raise artifacts.ArtifactError from exc

    # get catalog status
    current, to_update = _catalog_status(catalog_dict, context, policy=policy)
    catalog_status = 'present'
    if not catalog_dict:
        catalog_status = 'absent'
    elif to_update:
        catalog_status = 'stale'

    # update catalog if needed
    if to_update:
        catalog = manifest.build_catalog(
            to_update,
            original_catalog=current,
            mapped_grid_id=context.mapped_grid_id,
            source_image=context.source_image,
            source_label=context.source_label,
        )
        catalog_json = catalog.to_json_payload()
        ctrl.persist(catalog_json)
    else:
        catalog = current

    # ----- schema
    ctrl = SchemaCtrl(schema_fpath, policy)
    try:
        schema_dict = ctrl.fetch()
    except artifacts.ArtifactError as exc: # e.g., current schema corrupted
        raise artifacts.ArtifactError from exc

    sample_block = _sample(context.blocks_dir)
    schema_dict = manifest.build_schema(
        sample_block,
        original=schema_dict,
        mapped_grid_id=context.mapped_grid_id,
        sources=(context.source_image, context.source_label),
        label_color_map=context.label_color_map
    )
    ctrl.persist(schema_dict)

    return {
        'catalog_status': catalog_status,
        'catalog_updated': to_update,
        'cataloged_blocks_count': len(catalog),
        'schema_updated': True
    }

def _catalog_status(
    data_dict: dict[str, geo_core.CatalogEntry] | None,
    context: ManifestUpdateContext,
    *,
    policy: artifacts.LifecyclePolicy,
) -> tuple[geo_core.DataCatalog, list[str]]:
    '''Assess catalog status and determine required updates.'''

    # instantiate a catalog class from dict
    if data_dict:
        catalog = geo_core.DataCatalog.from_dict(data_dict)
    else:
        catalog = geo_core.DataCatalog() # empty catalog

    # get filenames from all current npz files in blks_dir
    blocks_dir = context.blocks_dir
    current = [f for f in os.listdir(blocks_dir) if f.endswith('npz')]
    if not current:
        raise FileNotFoundError('No block files found')

    # get filenames from the updated coordinates (parse from coords)
    updated = [f'{geo_utils.xy_name(c)}.npz' for c in context.updated_coords]

    # determine status
    cataloged = [os.path.basename(c['file_path']) for c in catalog.values()]
    catalog_status = {
        (True, False): 2,   # catalog present, no new blocks
        (True, True): 3,    # catalog present, has new blocks
        (False, False): 4,  # catalog absent, no new blocks
        (False, True): 5,   # catalog absent, has new blocks
    }[(bool(cataloged), bool(updated))]

    # policy choices
    match policy:
        # policy: build if missing
        case artifacts.LifecyclePolicy.BUILD_IF_MISSING:
            return catalog, {
                1: [f'{blocks_dir}/{f}' for f in current],
                2: [],
                3: [f'{blocks_dir}/{f}' for f in updated],
                4: [f'{blocks_dir}/{f}' for f in current],
                5: [f'{blocks_dir}/{f}' for f in current]
            }[catalog_status]
        # policy: force rebuild all
        case artifacts.LifecyclePolicy.REBUILD:
            return catalog, [f'{blocks_dir}/{f}' for f in current]
        # unsupported policy
        case _:
            raise NotImplementedError(f'Unsupported policy: {policy}')

def _sample(d: str) -> str:
    for f in os.listdir(d):
        if f.endswith('npz'):
            return f'{d}/{f}'
    raise ValueError(f'No .npz file found at {d}')
