'''doc'''

# standard import
import os
# local imports
import dataprep
import utils

def build_schema(
    data_cache_root: str,
    config: dataprep.DataprepConfigs
):
    '''doc'''

    # has test data flag
    has_test = os.path.exists(config['test_windows'])

    # read a sample block to get meta
    sample = next(iter(utils.load_json(config['train_blks']).values()))
    meta = dataprep.DataBlock().load(sample).meta

    # populate schema dict
    schema = {
        'schema_version': '1.0.0',

        'dataset': {
            'name': os.path.basename(data_cache_root), # dataset name
            'created_at': utils.get_timestamp(),
            'dataprep_commit': '',
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
            'ignore_index': meta['ignore_label']
        },

        'labels': {
            'label_num_classes': meta['label_num_classes'],
            'label_to_ignore': meta['label_to_ignore'],
            'heads_topology': _get_topology(meta['label_count'])
        },

        'normalization': {
            'method': 'per-channel-zscore',
            'fit_stats_file': _resolve(config['fit_img_stats']),
            'test_stats_file': _resolve(config['test_img_stats']),
        },

        'splits': {
            'train_blocks': _resolve(config['train_blks']),
            'val_blocks': _resolve(config['val_blks']),
            'test_blocks': _resolve(config['test_all_blks']) if has_test else {}
        },

        'training_stats': {
            'class_counts_train': _resolve(config['lbl_count_train']),
            'class_counts_global': _resolve(config['lbl_count_global'])
        }
    }

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
    return {'fpath': fpath, 'sh256': hash_records[fname]}

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