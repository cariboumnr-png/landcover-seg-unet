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
# local imports
import landseg.artifacts as artifacts
import landseg.geopipe.core as geo_core
import landseg.geopipe.foundation.data_blocks.manifest as manifest
import landseg.geopipe.utils as geo_utils
import landseg.utils as utils

# ------------------------------Public  Dataclass------------------------------
@dataclasses.dataclass
class ManifestUpdateContext:
    '''Context describing a manifest update operation.'''
    updated_coords: list[tuple[int, int]]   # grid coords for blocks that were created
    source_image: str               # path to the source image raster
    source_label: str | None        # optional path to the label raster
    mapped_grid_id: str             # id for the grid the blocks are mapped to

# -------------------------------Public Function-------------------------------
def update_manifest(
    context: ManifestUpdateContext,
    logger: utils.Logger,
    *,
    artifacts_dir: str,
    policy: artifacts.LifecyclePolicy
):
    '''
    Update dataset manifest artifacts according to lifecycle policy.

    This function manages coordinated updates to the dataset-level
    `catalog.json` and `schema.json` files. It determines the current
    state of cataloged block artifacts relative to blocks present on
    disk, applies the specified lifecycle policy to decide whether to
    rebuild or append entries, and writes updated JSON artifacts with
    integrity hashes.

    Args:
        context: A `ManifestUpdateContext` describing updated block
            coordinates and their source provenance.
        logger: Logger instance used to report status, decisions, and
            errors.
        artifacts_dir: Root dir containing `blocks/`, `catalog.json`,
            and `schema.json`.
        policy: Lifecycle policy controlling whether manifests are
            rebuilt unconditionally or only when stale.
    '''

    # blocks dir
    blk_dir = f'{artifacts_dir}/blocks'

    # load catalog JSON
    ctrl_args = (f'{artifacts_dir}/catalog.json', 'json', policy)
    ctrl = artifacts.Controller[dict[str, geo_core.CatalogEntry]](*ctrl_args)
    try:
        data = ctrl.fetch()
    except artifacts.ArtifactError as exc:
        logger.log('ERROR', f'Error loading {ctrl.fp}: {exc}')
        raise artifacts.ArtifactError from exc
    # action
    current, to_update = _catalog_status(data, context, blk_dir, policy, logger)
    if to_update:
        catalog = manifest.build_catalog(
            current,
            to_update,
            context.source_image,
            context.source_label,
            context.mapped_grid_id
        )
        catalog_json = catalog.to_json_payload()
        ctrl.persist(catalog_json)

    # load schema JSON
    ctrl_args = (f'{artifacts_dir}/schema.json', 'json', policy)
    ctrl = artifacts.Controller[geo_core.DataSchema](*ctrl_args)
    try:
        current_schema = ctrl.fetch()
    except artifacts.ArtifactError as exc:
        logger.log('ERROR', f'Error loading {ctrl.fp}: {exc}')
        raise artifacts.ArtifactError from exc
    # action
    sample_blk = f'{blk_dir}/{next(iter(os.listdir(blk_dir)))}'
    schema_dict = manifest.build_schema(
        current_schema,
        context.source_image,
        context.source_label,
        context.mapped_grid_id,
        sample_blk
    )
    ctrl.persist(schema_dict)

def _catalog_status(
    data_dict: dict[str, geo_core.CatalogEntry] | None,
    context: ManifestUpdateContext,
    blocks_dir: str,
    policy: artifacts.LifecyclePolicy,
    logger: utils.Logger,
) -> tuple[geo_core.DataCatalog, list[str]]:
    '''Assess catalog status and determine required updates.'''

    # instantiate a catalog class from dict
    if data_dict:
        catalog = geo_core.DataCatalog.from_dict(data_dict)
    else:
        catalog = geo_core.DataCatalog() # empty catalog

    # get filenames from all current npz files in blks_dir
    # this include new/updated blocks
    current = [f for f in os.listdir(blocks_dir) if f.endswith('npz')]
    if not current:
        raise FileNotFoundError('No block files found')
    logger.log('INFO', f'All block files on disk count: {len(current)}')

    # get filenames from the updated coordinates (parse from coords)
    updated = [f'{geo_utils.xy_name(c)}.npz' for c in context.updated_coords]
    logger.log('INFO', f'Updated block files to disk count: {len(updated)}')

    # determine status
    cataloged = [os.path.basename(c['file_path']) for c in catalog.values()]
    logger.log('INFO', f'Catalogued blocks files count: {len(cataloged)}')
    catalog_status = {
        (True, False): 2,   # catalog present, no new blocks
        (True, True): 3,    # catalog present, has new blocks
        (False, False): 4,  # catalog absent, no new blocks
        (False, True): 5,   # catalog absent, has new blocks
    }[(bool(cataloged), bool(updated))]
    logger.log('INFO', f'Cataloging status: {catalog_status}')

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
