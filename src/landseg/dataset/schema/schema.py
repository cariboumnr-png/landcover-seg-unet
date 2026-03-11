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
Schema builders for dataprep artifacts. Emits a dataset-wide JSON schema
from cached blocks and grid metadata, and can derive a minimal schema
from a single block (overfit mode).

Public APIs:
    - build_schema_full: Generate and write the dataset schema JSON.
    - build_schema_one_block: Build a minimal schema from one block.
'''

# standard import
import os
# local imports
import landseg.core as core
import landseg.dataset as dataset
import landseg.dataset.blockbuilder as blockbuilder
import landseg.utils as utils

# -------------------------------Public Function-------------------------------
def build_schema_full(
    world_grid: core.GridLayoutLike,
    data_cache_root: str,
    config: dataset.DataprepConfigs
) -> None:
    '''
    Generate and persist the dataset schema JSON from data and grid.

    Args:
        world_grid: Tuple of (grid_id, GridLayout) describing the target
            grid.
        data_cache_root: Root directory for the dataset cache;
            schema.json is written here.
        config: Consolidated dataprep configuration with input, output,
            and process fields.

    Raises:
        FileNotFoundError: If hash records for referenced artifacts are
            missing when resolving paths and checksums.

    Note: this function does not return a schema dict, but write one as
    JSON to disk.
    '''

    # has test data flag
    has_test = os.path.exists(config['test_windows'])
    test_data_has_label = os.path.exists(config['test_input_lbl'] or '')

    # read a sample block to get meta
    sample = next(iter(utils.load_json(config['train_blks']).values()))
    sample_data = blockbuilder.DataBlock().load(sample).data
    sample_meta = blockbuilder.DataBlock().load(sample).meta

    # image and label shape
    image_shape = sample_data.image_normalized.shape
    label_shape = sample_data.label_masked.shape

    # head topology
    parent, parent_cls = _get_topo(sample_meta['label_count'])

    # populate schema dict
    schema: core.SchemaFull = {
        'schema_version': '1.1',

        'dataset': {
            'name': os.path.basename(data_cache_root), # dataset name
            'created_at': utils.get_timestamp('%Y-%m-%dT%H:%M:%S'), # ISO-8601
            'dataprep_commit': 'dev', # to be fixed once branch stable
            'data_source': {
                'fit_image_path': config['fit_input_img'],
                'fit_label_path': config['fit_input_lbl'],
                'test_image_path': config['test_input_img'],
                'test_label_path': config['test_input_lbl'],
            },
            'has_test_data': has_test,
            'test_data_has_label': test_data_has_label
        },

        'world_grid': {
            'gid': world_grid.gid,
            'tile_size_x': world_grid.tile_size[0],
            'tile_size_y': world_grid.tile_size[1],
            'tile_overlap_x': world_grid.tile_overlap[0],
            'tile_overlap_y': world_grid.tile_overlap[1],
            'tile_step_x': world_grid.tile_size[0] - world_grid.tile_overlap[0],
            'tile_step_y': world_grid.tile_size[1] - world_grid.tile_overlap[1]
        },

        'io_conventions': {
            'block_format': 'npz',
            'keys': {
                'image_key': 'image_normalized',
                'label_key': 'label_masked',
                'valid_mask_key': 'valid_mask',
                'meta_key': 'meta'
            },
            'shapes': {
                'image_order': 'C,H,W',
                'label_order': 'L,H,W'
            },
            'dtypes': {
                'image': 'float32',
                'label': 'uint8',
                'valid_mask': 'bool'
            },
            'ignore_index': sample_meta['ignore_label']
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
            'label_num_classes': sample_meta['label_num_classes'],
            'label_to_ignore': sample_meta['label_to_ignore'],
            'head_parent': parent,
            'head_parent_cls': parent_cls,
        },

        'splits': {
            'train_blocks': config['train_blks'],
            'val_blocks': config['val_blks'],
            'test_blocks': config['test_valid_blks']
        },

        'training_stats': {
            'class_counts_train': config['lbl_count_train'],
            'class_counts_global': config['lbl_count_global']
        },

        'normalization': {
            'method': 'per-channel-zscore',
            'fit_stats_file': config['fit_img_stats'],
            'test_stats_file': config['test_img_stats'],
        },

        'checksums': {
            'train_blocks': _resolve(config['train_blks']),
            'val_blocks': _resolve(config['val_blks']),
            'test_blocks': _resolve(config['test_valid_blks']),
            'class_counts_train': _resolve(config['lbl_count_train']),
            'class_counts_global': _resolve(config['lbl_count_global']),
            'fit_stats_file': _resolve(config['fit_img_stats']),
            'test_stats_file': _resolve(config['test_img_stats'])
        }
    }

    # write schema to json
    utils.write_json(f'{data_cache_root}/schema.json', schema)

def build_schema_one_block(
    block_fpath: str,
    block: blockbuilder.DataBlock
) -> core.SchemaOneBlock:
    '''
    Build a minimal schema from a single block for overfit tests.

    Args:
        block_fpath: Filesystem path to the serialized block artifact.
        block: Loaded DataBlock instance corresponding to block_fpath.

    Returns:
        dict: Minimal schema including topology, splits, and per-head
            counts inferred from the provided block.
    '''

    data = block.data
    meta = block.meta
    counts = meta['label_count']
    cc = {k: [1] * len(counts[k]) for k in counts if k != 'original_label'}
    parent, parent_cls = _get_topo(counts)
    schema: core.SchemaOneBlock = {
        'dataset_name': meta['block_name'],
        'image_channel': data.image_normalized.shape[0],
        'ignore_index': meta['ignore_label'],
        'block_size': data.image_normalized.shape[1], # here assume H==W
        'class_counts': cc, # neutral
        'logit_adjust': {k: [1.0] * len(v) for k, v in cc.items()}, # neutral
        'head_parent': parent,
        'head_parent_cls': parent_cls,
        'train_split': {meta['block_name']: block_fpath},
        'val_split': {meta['block_name']: block_fpath}
    }
    return schema

# ------------------------------private  function------------------------------
def _resolve(fpath: str) -> str:
    '''Resolve an artifact path to its recorded SHA-256 in hash.json.'''

    # early exit if file does not exist
    if not os.path.exists(fpath):
        return ''

    # get file root and name
    root = os.path.dirname(fpath)
    fname = os.path.basename(fpath)

    # default hash record at root
    try:
        hash_records: dict[str, str] = utils.load_json(f'{root}/hash.json')
    except FileNotFoundError as e:
        raise e
    # sanity checks
    if not 'root' in hash_records and hash_records['root'] == root:
        raise ValueError('Hash records root not matching with input root')
    if fname not in hash_records:
        raise ValueError('File hash not in record')

    # return a dict
    return hash_records[fname]

def _get_topo(label_count: dict[str, list[int]]):
    '''Derive head topology (parent-child) from label count naming.'''

    head_parent: dict[str, str | None] = {}
    head_parent_cls: dict[str, int | None] = {}
    # iterate through label counts
    for layer_name in label_count:
        if layer_name == 'original_label': # skip this
            continue
        # emit topology for current convention - from layer naming
        if layer_name == 'layer1':
            head_parent[layer_name] = None
            head_parent_cls[layer_name] = None
        elif layer_name.startswith('layer2_'):
            cls_id = int(layer_name.split('layer2_')[1])
            head_parent[layer_name] = 'layer1'
            head_parent_cls[layer_name] = cls_id
        else:
            # if future names appear, one can decide to raise or set None
            head_parent[layer_name] = None
            head_parent_cls[layer_name] = None

    # return the dicts
    return head_parent, head_parent_cls
