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
Utilities for maintaining dataset catalog and metadata files.

This module provides functions to update or create `catalog.json` and
`metadata.json` for a block-structured dataset. It inspects block files,
records provenance information, and ensures catalog integrity during data
preparation workflows.
'''

# standard imports
import dataclasses
import os
# local imports
import landseg.geopipe.core as geo_core
import landseg.geopipe.utils as geo_utils
import landseg.utils as utils

T_FORMAT = '%Y-%m-%dT%H:%M:%S'  # ISO-8601

# ------------------------------Public  Dataclass------------------------------
@dataclasses.dataclass
class CatalogUpdateContext:
    '''Context object describing a catalog update operation.'''
    updated_coords: list[tuple[int, int]]   # grid coords for blocks that were created
    source_image: str               # path to the source image raster
    source_label: str | None        # optional path to the label raster
    mapped_grid_id: str             # id for the grid the blocks are mapped to

# -------------------------------Public Function-------------------------------
def update_catalog(
    input_block_fpaths: list[str],
    update_context: CatalogUpdateContext,
    current_catalog: geo_core.BlocksCatalog,
) -> geo_core.BlocksCatalog:

    '''
    Create or update the dataset's `catalog.json`.

    Behavior:
    - Loads the existing catalog if present; otherwise starts a new one.
    - Determines which block files (`*.npz`) to be cataloged based on:
        * provided block coordinates, or
        * scanning the `/blocks` directory if no coordinates exist.
    - For each block:
        * loads metadata from disk
        * computes file-level SHA-256
        * records source image/label provenance
        * stores spatial indices and basic statistics

    Args:
        updated_blocks: A `CatalogUpdateContext` describing new or
            updated blocks and their source provenance.
        root_dir: Root directory of the dataset, containing `blocks/` and
            optionally `catalog.json`.
        logger: Logger used to report progress and warnings.

    Raises:
        FileNotFoundError: If `/blocks/` is missing or no block files are
            found when attempting to create a new catalog.
    '''

    # get hash values from input rasters
    img_hash = utils.hash_artifacts(update_context.source_image, False)
    if update_context.source_label:
        lbl_hash = utils.hash_artifacts(update_context.source_label, False)
    else:
        lbl_hash = None

    # add to current catalog dict
    for fp in input_block_fpaths:
        meta = geo_core.DataBlock.load(fp).meta
        row, col = geo_utils.name_xy(meta['block_name'])
        current_catalog[(col, row)] = {
            'block_name': meta['block_name'],
            'file_path': fp,
            'row_col': [row, col],
            'base_valid_px': meta['valid_ratios']['base'],
            'base_class_count': meta['label_count']['base'],
            'schema_version': '1.0.0',
            'creation_time': utils.get_file_ctime(fp, T_FORMAT),
            'sha_256': utils.hash_artifacts(fp, False),
            'aligned_grid': update_context.mapped_grid_id,
            'source_image': update_context.source_image,
            'source_image_sha_256': img_hash,
            'source_label': update_context.source_label,
            'source_label_sha_256': lbl_hash,
        }
    return current_catalog

def update_meta(
    updated_blocks: CatalogUpdateContext,
    root_dir: str,
    logger: utils.Logger
) -> None:
    '''
    Create or update the dataset-level `metadata.json`.

    Behavior:
    - If a metadata file exists:
        * updates `last_updated`
        * appends new grid IDs and data sources
        * rehashes the metadata file after writing
    - If not:
        * inspects a sample block to infer shapes, dtypes, and label info
        * builds a full `CatalogMeta` structure from scratch
        * writes and hashes the new metadata file

    Args:
        updated_blocks: Context with grid ID and source raster paths.
        root_dir: Dataset directory containing `blocks/` and metadata
            targets.
        logger: Logger for status messages.
    '''

    # parse
    grid_id = updated_blocks.mapped_grid_id
    src_img = updated_blocks.source_image
    src_lbl = updated_blocks.source_label

    # try load the current meta dict
    meta_fpath = f'{root_dir}/metadata.json'
    try:
        meta_dict = utils.load_json(meta_fpath)
        meta_dict['dataset']['last_updated'] = utils.get_timestamp(T_FORMAT)
        if not grid_id in meta_dict['dataset']['mapped_grids']:
            meta_dict['dataset']['mapped_grids'].append(grid_id)
        if not src_img in meta_dict['dataset']['data_source']['image_paths']:
            meta_dict['dataset']['data_source']['image_paths'].append(src_img)
        if src_lbl and not src_lbl in meta_dict['dataset']['data_source']['label_paths']:
            meta_dict['dataset']['data_source']['label_paths'].append(src_lbl)
        # save and add hash to record
        utils.write_json(meta_fpath, meta_dict)
        utils.hash_artifacts(meta_fpath)
        return
    #
    except FileNotFoundError:
        logger.log('INFO', 'Metadata JSON not found, create one')

    # read a sample block to required info
    sample_file = next(iter(os.listdir(f'{root_dir}/blocks')))
    sample_blk = geo_core.DataBlock.load(f'{root_dir}/blocks/{sample_file}')
    # image and label shape
    image_shape = sample_blk.data.image.shape
    label_shape = sample_blk.data.label_stack.shape

    # create new
    meta_dict: geo_core.CatalogMeta = {
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
                'shape': [*sample_blk.data.image.shape],
                'C': image_shape[0],
                'H': image_shape[1],
                'W': image_shape[2]
            },
            'label': {
                'order': 'L,H,W',
                'shape': [*sample_blk.data.label_stack.shape],
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
