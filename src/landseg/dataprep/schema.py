'''doc'''

# standard import
import os
# local imports
import landseg.dataprep as dataprep
import landseg.dataprep.blockbuilder as blockbuilder
import landseg.grid as grid
import landseg.utils as utils

def build_schema(
    world_grid: tuple[str, grid.GridLayout],
    data_cache_root: str,
    config: dataprep.DataprepConfigs
):
    '''doc'''

    # has test data flag
    has_test = os.path.exists(config['test_windows'])

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
        'schema_version': '1.0.0',

        'dataset': {
            'name': os.path.basename(data_cache_root), # dataset name
            'created_at': utils.get_timestamp('%Y-%m-%dT%H:%M:%S'), # ISO-8601
            'has_test_data': has_test,
            'dataprep_commit': 'dev', # to be fixed once branch stable
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
            'test_stats_file': _resolve(config['test_img_stats']) if has_test else {},
        },

        'splits': {
            'train_blocks': _resolve(config['train_blks']),
            'val_blocks': _resolve(config['val_blks']),
            'test_blocks': _resolve(config['test_valid_blks']) if has_test else {}
        },

        'training_stats': {
            'class_counts_train': _resolve(config['lbl_count_train']),
            'class_counts_global': _resolve(config['lbl_count_global'])
        }
    }

    # write schema to json
    utils.write_json(f'{data_cache_root}/schema.json', schema)

def _resolve(fpath: str) -> dict[str, str]:
    '''doc'''

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
    '''doc'''

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
