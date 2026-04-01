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

'''Data blocks artifacts lifecycle management.'''

# standard imports
import os
# local imports
import landseg.geopipe.artifacts as artifacts
import landseg.geopipe.core as geo_core
import landseg.geopipe.foundation.data_blocks as data_blocks
import landseg.geopipe.utils as geo_utils
import landseg.utils as utils

# -------------------------------Public Function-------------------------------
def prepare_mapped_raster_windows(
    world_grid: geo_core.GridLayout,
    image_label_fpaths: tuple[str, str],
    logger: utils.Logger,
    *,
    artifacts_dir: str,
    policy: artifacts.LifecyclePolicy
) -> data_blocks.MappedRasterWindows:
    '''doc'''

    # aliases
    gid = world_grid.gid
    img, lbl = image_label_fpaths

    # policy: build if missing
    if policy is artifacts.LifecyclePolicy.BUILD_IF_MISSING:
        logger.log('INFO', f'Try to load mapped windows from {gid}')
        try:
            windows = data_blocks.load_mapped_windows(gid, artifacts_dir)
            logger.log('INFO', f'Mapped windows from {gid} loaded')
            return windows
        except FileNotFoundError:
            logger.log('INFO', f'Mapped windows from {gid} not found')
            windows = data_blocks.map_rasters(world_grid, img, lbl, logger)
            data_blocks.save_mapped_windows(gid, windows, artifacts_dir)
            logger.log('INFO', f'Mapped windows from {gid} created')
            return windows

    # unsupported policy
    else:
        msg = f'Currently unsupported policy: {policy}'
        logger.log('ERROR', msg)
        raise NotImplementedError(msg)

def update_blocks_catalog(
    context: data_blocks.CatalogUpdateContext,
    logger: utils.Logger,
    *,
    artifacts_dir: str,
    policy: artifacts.LifecyclePolicy
) -> None:
    '''doc'''

    # alises
    blks_dir = f'{artifacts_dir}/blocks'
    catalog_fpath = f'{artifacts_dir}/catalog.json'

    # early exit if not data blocks are present
    if not os.path.exists(blks_dir):
        logger.log('ERROR', f'No /blocks/ folder at {artifacts_dir}')
        raise FileNotFoundError

    # cataloged vs current
    catalog = geo_core.BlocksCatalog.from_json(catalog_fpath)
    cataloged = [os.path.basename(c['file_path']) for c in catalog.values()]
    current = [f for f in os.listdir(blks_dir) if f.endswith('npz')]
    # here assert the two are the same FOR NOW
    if cataloged:
        assert set(cataloged) == set(current), f'{len(cataloged), len(current)}'

    # updated blocks canonical filenames
    updated = [f'{geo_utils.xy_name(c)}.npz' for c in context.updated_coords]

    # status assesment
    flag = 0
    if catalog:
        if not updated:
            m = f'Current entries: {len(catalog)}; no new blcoks'
            logger.log('INFO', m)
            flag = 1
        else:
            m = f'Current entries: {len(catalog)}; new blcoks: {len(updated)}'
            logger.log('INFO', m)
            flag = 2
    else:
        if not updated:
            if not current:
                m = 'Catalog absent; no new blocks; no exisiting blocks'
                logger.log('ERROR', m)
                raise FileNotFoundError
            m = f'Catalog absent; no new blocks; exisiting blocks: {len(current)}'
            logger.log('INFO', m)
            flag = 3
        else:
            m = f'Catalog absent; new blocks: {len(updated)}'
            logger.log('INFO', m)
            flag = 4

    # policy: rebuild if stale (includes build if missing)
    if policy is artifacts.LifecyclePolicy.REBUILD_IF_STALE:
        fnames = {1: [], 2: updated, 3: current, 4: updated}[flag]
    # policy: force rebuild
    elif policy is artifacts.LifecyclePolicy.REBUILD:
        fnames = {1: current, 2: updated + current, 3: current, 4: updated}[flag]
    # unsupported policy
    else:
        m = f'Currently unsupported policy: {policy}'
        logger.log('ERROR', m)
        raise NotImplementedError(m)

    # take action and save
    if fnames:
        fpaths = [f'{blks_dir}/{f}' for f in set(fnames)]
        catalog = data_blocks.update_catalog(fpaths, context, catalog)
        catalog.save_json(catalog_fpath)
