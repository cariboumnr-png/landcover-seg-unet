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

T_FORMAT = '%Y-%m-%dT%H:%M:%S'  # ISO-8601

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
        fnames = [f'{core.yx_name(c)}.npz' for c in coords]
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
            fnames = [f'{core.yx_name(c)}.npz' for c in coords]

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
        row, col = core.name_yx(meta['block_name'])
        catalog[(col, row)] = {
            'block_name': meta['block_name'],
            'file_path': fp,
            'row_col': [row, col],
            'valid_px': meta['valid_ratios']['base'],
            'class_count': meta['label_count']['base'],
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

def update_meta(
    updated_blocks: CatalogUpdateContext,
    root_dir: str,
    logger: utils.Logger
) -> None:
    '''Create/update catalog meta JSON.'''

    # parse
    grid_id = updated_blocks.mapped_grid_id
    src_img = updated_blocks.source_image
    src_lbl = updated_blocks.source_label

    # try load the current meta dict
    meta_fpath = f'{root_dir}/metadata.json'
    try:
        meta_dict = utils.load_json(meta_fpath)
        meta_dict['dataset']['last_updated'] = utils.get_timestamp(T_FORMAT)
        meta_dict['dataset']['mapped_grids'].append(grid_id)
        meta_dict['dataset']['data_source']['image_paths'].append(src_img)
        if src_lbl:
            meta_dict['dataset']['data_source']['label_paths'].append(src_lbl)
        # save and add hash to record
        utils.write_json(meta_fpath, meta_dict)
        utils.hash_artifacts(meta_fpath)
        return
    #
    except FileNotFoundError:
        logger.log('INFO', 'Metadata JSON not found, create one')

    # read a sample block to required info
    sample = next(iter(os.listdir(f'{root_dir}/blocks')))
    sample_blk = core.DataBlock.load(f'{root_dir}/blocks/{sample}')
    # image and label shape
    image_shape = sample_blk.data.image.shape
    label_shape = sample_blk.data.label_stack.shape

    # create new
    meta_dict: core.CatalogMeta = {
        'dataset': {
            'name': os.path.basename(root_dir), # dataset name
            'last_updated': utils.get_timestamp(T_FORMAT),
            'dataprep_commit': 'dev', # to be fixed once branch stable
            'mapped_grids': [grid_id],
            'data_source': {
                'image_paths': [src_img],
                'label_paths': [src_lbl] if src_lbl else [],
            },
        },

        'io_conventions': {
            'block_format': 'npz',
            'shapes': {
                'image_order': 'C,H,W',
                'label_order': 'L,H,W'
            },
            'dtypes': {
                'image': 'float32',
                'label': 'uint8',
            },
            'ignore_index': sample_blk.meta['ignore_index']
        },

        'tensor_shapes': {
            'image': {
                'order': 'C,H,W',
                'shape': [image_shape[0], image_shape[1], image_shape[2]],
                'C': image_shape[0],
                'H': image_shape[1],
                'W': image_shape[2]
            },
            'label': {
                'order': 'L,H,W',
                'shape': [label_shape[0], label_shape[1], label_shape[2]],
                'L': label_shape[0],
                'H': label_shape[1],
                'W': label_shape[2]
            }
        },

        'labels': {
            'label_num_classes': sample_blk.meta['label_num_cls'],
            'label_to_ignore': sample_blk.meta['label_ignore_cls'],
            'channel_parent': sample_blk.meta['label_ch_parent'],
            'channel_parent_cls': sample_blk.meta['label_ch_parent_cls'],
        },
    }
    # save and add hash to record
    utils.write_json(meta_fpath, meta_dict)
    utils.hash_artifacts(meta_fpath)
    return
