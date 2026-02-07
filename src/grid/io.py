'''
I/O utilites for class `GridLayout`.
'''

# standard imports
import hashlib
import json
import os
# local imports
import grid
import utils

def save_grid(
    grid_name: str,
    grid_obj: grid.GridLayout,
    dirpath: str
) -> None:
    '''doc'''

    # prepare output dir
    os.makedirs(dirpath, exist_ok=True)

    # get grid object payload and pickle
    payload = grid_obj.to_payload()
    utils.write_pickle(f'{dirpath}/{grid_name}.pkl', payload)

    # write meta to json
    meta = {
        'schema_id': getattr(grid_obj, 'SCHEMA_ID', 'unknown'),
        'sha256': _hash_payload(payload),
        'mode': payload['mode'],
        'spec': payload['spec'],
        'extent': payload['extent'],
        'tiles count': len(payload['windows'])
    }
    utils.write_json(f'{dirpath}/{grid_name}_meta.json', meta)

def load_grid(
    grid_name: str,
    dirpath: str
) -> grid.GridLayout:
    '''doc'''

    # load payload and meta json
    payload = utils.load_pickle(f'{dirpath}/{grid_name}.pkl')
    meta = utils.load_json(f'{dirpath}/{grid_name}_meta.json')

    # check hash
    if _hash_payload(payload) != meta['sha256']:
        raise ValueError(f'Grid {grid_name} might be altered/damaged.')

    # schema guard
    expected = grid.GridLayout.SCHEMA_ID
    found = meta.get('schema_id', None)
    if found != expected:
        raise ValueError(f'Unsupported schema: {found}; expected {expected}.')

    # otherwise return class object via class method
    return grid.GridLayout.from_payload(payload)

def _hash_payload(payload: grid.GridLayoutPayload) -> str:
    '''Hash payload.'''

    canonical = {
        'mode': payload['mode'],
        'spec': payload['spec'],
        'extent': payload['extent'],
        'windows': [
            (k[0], k[1], w.col_off, w.row_off, w.width, w.height)
            for k, w in sorted(payload['windows'].items())
        ]
    }
    blob = json.dumps(canonical, sort_keys=True, separators=(',', ':'))
    sha256 = hashlib.sha256(blob.encode()).hexdigest()
    return sha256
