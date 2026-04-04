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
Data block catalog and metadata artifact lifecycle management.

This module orchestrates creation, validation, and incremental updates of
dataset manifest artifacts, including `catalog.json` and `metadata.json`.
It coordinates policy-driven rebuilds, detects newly created or modified
data blocks, and ensures consistency between on-disk block artifacts and
their recorded metadata throughout data preparation and update workflows.
'''

# standard imports
import dataclasses
import os
# local imports
import landseg.geopipe.artifacts as artifacts
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
    `catalog.json` and `metadata.json` files. It determines the current
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
            and `metadata.json`.
        policy: Lifecycle policy controlling whether manifests are
            rebuilt unconditionally or only when stale.
    '''

    # aliases
    blk_dir = f'{artifacts_dir}/blocks'
    cata_fp = f'{artifacts_dir}/catalog.json'
    meta_fp = f'{artifacts_dir}/metadata.json'

    # catalog JSON
    catalog, to_add = _catalog_status(context, blk_dir, cata_fp, policy, logger)
    if to_add:
        _catalog = manifest.build_catalog(
            catalog,
            to_add,
            context.source_image,
            context.source_label,
            context.mapped_grid_id
        )
        catalog_json = _catalog.to_json_payload()
        artifacts.write_json_hash(cata_fp, catalog_json)

    # metadata JSON
    sample_blk = f'{blk_dir}/{next(iter(os.listdir(blk_dir)))}'
    current_metadata = _metadata_status(meta_fp, policy, logger)
    meta_dict = manifest.build_metadata(
        current_metadata,
        context.source_image,
        context.source_label,
        context.mapped_grid_id,
        sample_blk
    )
    artifacts.write_json_hash(meta_fp, meta_dict)

def _catalog_status(
    context: ManifestUpdateContext,
    blocks_dir: str,
    catalog_fpath: str,
    policy: artifacts.LifecyclePolicy,
    logger: utils.Logger,
) -> tuple[geo_core.DataCatalog, list[str]]:
    '''Assess catalog status and determine required updates.'''

    # get filenames from all current npz files in blks_dir
    # this include new/updated blocks
    current = [f for f in os.listdir(blocks_dir) if f.endswith('npz')]
    if not current:
        raise FileNotFoundError('No block files found')
    logger.log('INFO', f'All block files on disk count: {len(current)}')

    # get filenames from the updated coordinates (parse from coords)
    updated = [f'{geo_utils.xy_name(c)}.npz' for c in context.updated_coords]
    logger.log('INFO', f'Updated block files to disk count: {len(updated)}')

    # load catalog json and give it a status
    catalog_dict: dict[str, geo_core.CatalogEntry] # type declaration
    load_status, m, catalog_dict = artifacts.load_json_hash(catalog_fpath)
    if load_status: # non-zero status indicates false catalog.json -> rebuild
        catalog = geo_core.DataCatalog() # empty
        cataloged = None
        catalog_status = 1
        logger.log('INFO', f'Catalog JSON loading error: {m}')
    else:
        # get filenames from the catalog
        catalog = geo_core.DataCatalog.from_dict(catalog_dict)
        cataloged = [os.path.basename(c['file_path']) for c in catalog.values()]
        logger.log('INFO', f'Catalogued blocks files count: {len(cataloged)}')
        # determine status
        catalog_status = {
            (True, False): 2,   # catalog present, no new blocks
            (True, True): 3,    # catalog present, has new blocks
            (False, False): 4,  # catalog absent, no new blocks
            (False, True): 5,   # catalog absent, has new blocks
        }[(bool(cataloged), bool(updated))]
    logger.log('INFO', f'Cataloging status: {catalog_status}')

    # policy: rebuild if stale (includes build if missing)
    if policy is artifacts.LifecyclePolicy.REBUILD_IF_STALE:
        return catalog, {
            1: [f'{blocks_dir}/{f}' for f in current],
            2: [],
            3: [f'{blocks_dir}/{f}' for f in updated],
            4: [f'{blocks_dir}/{f}' for f in current],
            5: [f'{blocks_dir}/{f}' for f in current]
        }[catalog_status]
    # policy: force rebuild all
    if policy is artifacts.LifecyclePolicy.REBUILD:
        return catalog, [f'{blocks_dir}/{f}' for f in current]
    # unsupported policy
    m = f'Currently unsupported policy: {policy}'
    logger.log('ERROR', m)
    raise NotImplementedError(m)

def _metadata_status(
    metadata_fpath: str,
    policy: artifacts.LifecyclePolicy,
    logger: utils.Logger,
) -> geo_core.DataSchema | None:
    '''Load and evaluate dataset metadata state.'''

    # load metadict
    original_meta: geo_core.DataSchema | None
    load_status, m, original_meta = artifacts.load_json_hash(metadata_fpath)
    if load_status: # non-zero status indicates false metadata.json -> rebuild
        original_meta = None
        logger.log('INFO', f'Metadata JSON loading error: {m}')
    else:
        logger.log('INFO', 'Metadata JSON loading successful')

    # policy: rebuild if stale (includes build if missing)
    if policy is artifacts.LifecyclePolicy.REBUILD_IF_STALE:
        return original_meta # let the builder decide whether to update
    # policy: force rebuild all
    if policy is artifacts.LifecyclePolicy.REBUILD:
        return None
    # unsupported policy
    m = f'Currently unsupported policy: {policy}'
    logger.log('ERROR', m)
    raise NotImplementedError(m)
