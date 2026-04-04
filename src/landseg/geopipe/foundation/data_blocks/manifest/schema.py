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
Utilities for maintaining dataset-level catalog and schema files.

This module defines helper routines for creating and updating
``schema.json`` for block-structured geospatial datasets. It derives
dataset-wide properties by inspecting stored block artifacts, records
provenance links to source imagery and labels, tracks grid alignment
history, and standardizes I/O conventions required for reproducible data
preparation and downstream model consumption.
'''

# local imports
import landseg.geopipe.core as geo_core
import landseg.utils as utils

T_FORMAT = '%Y-%m-%dT%H:%M:%S'  # ISO-8601

# -------------------------------Public Function-------------------------------
def build_schema(
    original: geo_core.DataSchema | None,
    source_image: str,
    source_label: str | None,
    mapped_grid_id: str,
    sample_block_fpath: str
) -> geo_core.DataSchema:
    '''
    Create or update the dataset-level `schema.json`.

    Manages global dataset schema describing data sources, spatial grids,
    tensor conventions, and label semantics. When existing schema is
    provided, it updates timestamps and appends new grid or source
    references while preserving all previously recorded structure. When
    no schema exists, it inspects a representative sample block to infer
    tensor shapes, data types, and label configuration, and constructs a
    complete schema specification from scratch.

    Args:
        original_meta: Existing `BlocksMetadata` object if present,
            otherwise `None` when creating new schema.
        source_image: File path or identifier of the source image used
            to generate dataset blocks.
        source_label: File path or identifier of the source label data,
            if applicable. May be `None` for image-only datasets.
        mapped_grid_id: Identifier of the spatial grid to which blocks
            are aligned.
        sample_block_fpath: File path to a representative block artifact
            used to infer dataset-wide shapes, dtypes, and label settings
            when initializing schema.

    Returns:
        A fully populated `BlocksMetadata` object reflecting updated or
        newly created dataset schema.
    '''

    # update route
    if original:
        # aliases
        grids = original['dataset']['mapped_grids']
        images = original['dataset']['data_source']['image_paths']
        labels = original['dataset']['data_source']['label_paths']
        # update
        original['dataset']['last_updated'] = utils.get_timestamp(T_FORMAT)
        if not mapped_grid_id in grids:
            grids.append(mapped_grid_id)
        if not source_image in images:
            images.append(source_image)
        if source_label and not source_label in labels:
            labels.append(source_label)
        return original

    # read from the sample block
    sample_blk = geo_core.DataBlock.load(sample_block_fpath)
    image_shape = sample_blk.data.image.shape
    label_shape = sample_blk.data.label_stack.shape

    # create route
    new: geo_core.DataSchema = {
        'schema_id': geo_core.foundation_data_schema.SCHEMA_ID,
        'dataset': {
            'name': '', # TBD
            'last_updated': utils.get_timestamp(T_FORMAT),
            'dataprep_commit': 'dev', # to be fixed once branch stable
            'mapped_grids': [mapped_grid_id],
            'data_source': {
                'image_paths': [source_image],
                'label_paths': [source_label] if source_label else [],
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
    return new
