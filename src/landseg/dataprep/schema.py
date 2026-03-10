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
    - build_schema: Generate and write the dataset schema JSON.
    - schema_from_a_block: Build a minimal schema from one block.
'''

# standard import
import os
import typing
# local imports
import landseg.dataprep as dataprep
import landseg.dataprep.blockbuilder as blockbuilder
import landseg.grid as grid
import landseg.utils as utils

# -------------------------------Public Function-------------------------------
def build_schema(
    world_grid: tuple[str, grid.GridLayout],
    data_cache_root: str,
    config: dataprep.DataprepConfigs
):
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
    '''

    # has test data flag
    has_test = os.path.exists(config['test_windows'])
    test_data_has_label = os.path.exists(config['test_input_lbl'] or '')

    # parse grid
    gid, work_grid = world_grid

    # read a sample block to get meta
    sample = next(iter(utils.load_json(config['train_blks']).values()))
    sample_data = blockbuilder.DataBlock().load(sample).data
    sample_meta = blockbuilder.DataBlock().load(sample).meta

    # image and label shape
    image_shape = sample_data.image_normalized.shape
    label_shape = sample_data.label_masked.shape

    # populate schema dict
    schema = {
        'schema_version': '1.0.1',

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
            'gid': gid,
            'tile_size_x': work_grid.tile_size[0],
            'tile_size_y': work_grid.tile_size[1],
            'tile_overlap_x': work_grid.tile_overlap[0],
            'tile_overlap_y': work_grid.tile_overlap[1],
            'tile_step_x': work_grid.tile_size[0] - work_grid.tile_overlap[0],
            'tile_step_y': work_grid.tile_size[1] - work_grid.tile_overlap[1]
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
            'heads_topology': _get_topology(sample_meta['label_count'])
        },

        'normalization': {
            'method': 'per-channel-zscore',
            'fit_stats_file': _resolve(config['fit_img_stats']),
            'test_stats_file': _resolve(config['test_img_stats'])
            if has_test else {},
        },

        'splits': {
            'train_blocks': _resolve(config['train_blks']),
            'val_blocks': _resolve(config['val_blks']),
            'test_blocks': _resolve(config['test_valid_blks'])
            if has_test else {}
        },

        'training_stats': {
            'class_counts_train': _resolve(config['lbl_count_train']),
            'class_counts_global': _resolve(config['lbl_count_global'])
        }
    }

    # write schema to json
    utils.write_json(f'{data_cache_root}/schema.json', schema)

def schema_from_a_block(
    block_fpath: str,
    block: blockbuilder.DataBlock
) -> dict[str, typing.Any]:
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
    return {
        'dataset_name': meta['block_name'],
        'image_channel': data.image_normalized.shape[0],
        'ignore_index': meta['ignore_label'],
        'block_size': data.image_normalized.shape[1], # here H==W
        'class_counts': cc,
        'logit_adjust': {k: [1.0] * len(v) for k, v in cc.items()},
        'topology': _get_topology(meta['label_count']),
        'train_split': {meta['block_name']: block_fpath},
        'val_split': {meta['block_name']: block_fpath}
    }

# ------------------------------private  function------------------------------
def _resolve(fpath: str) -> dict[str, str]:
    '''Resolve an artifact path to its recorded SHA-256 in hash.json.'''

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
    return {'fpath': fpath, 'sha256': hash_records[fname]}

def _get_topology(label_count: dict[str, list[int]]):
    '''Derive head topology (parent-child) from label count naming.'''

    topology: dict[str, dict[str, str | int | None]] = {}
    # iterate through label counts
    for layer_name in label_count:
        if layer_name == 'original_label': # skip this
            continue
        # emit topology for current convention - from layer naming
        if layer_name == 'layer1':
            topology[layer_name] = {'parent': None, 'parent_cls': None}
        elif layer_name.startswith('layer2_'):
            cls_id = int(layer_name.split('layer2_')[1])
            topology[layer_name] = {'parent': 'layer1', 'parent_cls': cls_id}
        else:
            # if future names appear, one can decide to raise or set None
            topology[layer_name] = {'parent': None, 'parent_cls': None}

    # return the dicts
    return topology
