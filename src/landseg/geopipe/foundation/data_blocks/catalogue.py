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
Maintain catalog JSON at given location.
'''

# standard imports
import dataclasses
import os
# local imports
import landseg.geopipe.core as core
import landseg.utils as utils

# ------------------------------Public  Dataclass------------------------------
@dataclasses.dataclass
class CatalogUpdateContext:
    '''Context for catalog updates.'''
    coords: list[tuple[int, int]]
    source_image: str
    source_label: str | None
    mapped_grid_id: str

# -------------------------------Public Function-------------------------------
def update_catalog(
    updated_blocks: CatalogUpdateContext,
    root_dir: str,
    logger: utils.Logger
) -> None:
    '''
    Update/create `catalog.json` to all valid block files on disk.

    Any blocks found on disk but missing from the catalog are hashed,
    loaded for metadata, and appended as new catalog entries.
    '''

    # check current directory
    blks_dir = f'{root_dir}/blocks'
    if not os.path.exists(blks_dir):
        logger.log('ERROR', f'No /blocks/ folder at {root_dir}')
        raise FileNotFoundError

    # load catalog.json
    catalog_path = f'{root_dir}/catalog.json'
    catalog = core.BlocksCatalog.from_json(catalog_path) # empty {} if no file

    # conditions
    coords = updated_blocks.coords
    if catalog:
        logger.log('INFO', f'Found {len(catalog)} existing catalog entries')
        if not coords:
            logger.log('INFO', 'No new blocks provided, exit')
            return
        logger.log('INFO', f'Update {len(coords)} new blocks')
        fnames = [f'{_yx_name(c)}.npz' for c in coords]
    else:
        logger.log('INFO', f'catalog.json not found at {root_dir}')
        if not coords:
            logger.log('INFO', 'No new blocks provided, scan existing blocks')
            fnames = [p for p in os.listdir(blks_dir) if p.endswith('npz')]
            if not fnames:
                logger.log('ERROR', f'No blocks found at {blks_dir}')
                raise FileNotFoundError
            logger.log('INFO', f'Found {len(fnames)} existing blocks, create')
        else:
            logger.log('INFO', f'Creat from {len(coords)} new blocks')
            fnames = [f'{_yx_name(c)}.npz' for c in coords]

    # get hash values from input rasters
    img_hash = utils.hash_artifacts(updated_blocks.source_image, False)
    if updated_blocks.source_label:
        lbl_hash = utils.hash_artifacts(updated_blocks.source_label, False)
    else:
        lbl_hash = None

    logger.log('INFO', 'Creating/updating catalog.json...')
    # add to current catalog dict
    for name in fnames:
        fp = f'{blks_dir}/{name}'
        meta = core.DataBlock.load(fp).meta
        row, col = _name_yx(meta['block_name'])
        catalog[(col, row)] = {
            'block_name': meta['block_name'],
            'file_path': fp,
            'row_col': [row, col],
            'valid_px': meta['valid_ratios']['layer1'],
            'class_count': meta['label_count']['layer1'],
            'schema_version': '1.0.0',
            'creation_time': utils.get_file_ctime(fp, '%Y-%m-%dT%H:%M:%S'),
            'sha_256': utils.hash_artifacts(fp, False),
            'aligned_grid': updated_blocks.mapped_grid_id,
            'source_image': updated_blocks.source_image,
            'source_image_sha_256': img_hash,
            'source_label': updated_blocks.source_label,
            'source_label_sha_256': lbl_hash,
        }
    catalog.save_json(catalog_path)

# coords <-> name helpers
def _yx_name(coords: tuple[int, int]) -> str:
    '''Convert (row, col) to a canonical block name string.'''

    # e.g., (12, 34) -> row_000012_col_000034
    y, x = coords
    return f'row_{y:06d}_col_{x:06d}'

def _name_yx(name: str) -> tuple[int, int]:
    '''Convert a canonical block name back to (row, col).'''

    # e.g.,  row_000012_col_000034 -> (12, 34)
    split = name.split('_')
    y_str, x_str = split[1], split[3]
    return int(y_str), int(x_str)
