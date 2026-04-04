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
I/O utilites for class `GridLayout`.

This module handles persistence of `GridLayout` objects via a pickled
payload and a JSON metadata sidecar. The payload represents a stable
world-grid indexed by pixel-origin coordinates (x_px, y_px). Raster
alignment is not stored and is applied at access time.

A schema identifier and a canonical hash are used to validate payload
compatibility and integrity on load.
'''

# standard imports
import os
# local imports
import landseg.geopipe.artifacts as artifacts
import landseg.geopipe.core as geo_core

# -------------------------------Public Function-------------------------------
def save_grid(
    grid_obj: geo_core.GridLayout,
    dirpath: str
) -> None:
    '''
    Serialize a `GridLayout` to disk.

    Writes a pickled payload and a JSON metadata file containing schema
    identifier, integrity hash, and summary fields. The payload encodes
    the world grid in stable pixel-origin space; no raster alignment
    state is persisted.

    Args:
        grid_name: Name of the grid in both `.pkl` and `.json` artifacts.
        grid_obj: The `GridLayout` instance to be saved.
        dirpath: Directory where artifacts are to be saved.
    '''

    # prepare output dir
    os.makedirs(dirpath, exist_ok=True)

    # get grid object payload and pickle
    payload = grid_obj.to_payload()
    # get gid as file name
    name = payload['artifact_meta']['gid']
    # get data (pop)
    data = payload.pop('data')
    # write domain tiles dict and write to JSON
    artifacts.write_pickle_hash(f'{dirpath}/{name}.pkl', data)
    # write meta dict and write to json
    artifacts.write_json_hash(f'{dirpath}/{name}_meta.json',payload)

def load_grid(
    grid_id: str,
    dirpath: str
) -> tuple[int, str, geo_core.GridLayout | None]:
    '''
    Load a `GridLayout` from disk.

    Validates payload hash and schema identifier before reconstructing
    the `GridLayout` instance. Raises if the payload is corrupted or the
    schema is unsupported.

    Args:
        grid_name: Name of the grid in both `.pkl` and `.json` artifacts.
        dirpath: Directory where artifacts are located.
    '''

    # load payload and meta json
    grid_data_path = f'{dirpath}/{grid_id}.pkl'
    meta_path = f'{dirpath}/{grid_id}_meta.json'
    # types declaration
    grid_status, grid_msg, tiles = artifacts.load_pickle_hash(grid_data_path)
    meta_status, meta_msg, meta = artifacts.load_json_hash(meta_path)

    # loading status
    status = {
        (False, False): 0,
        (True, False): 1,
        (False, True): 2,
        (True, True): 3
    }[bool(grid_status), bool(meta_status)]

    # combined summary message
    msg = {
        0: 'Grid .pkl and domain metadata JSON loaded successfully',
        1: f'Error loading Grid .pkl: {grid_msg}',
        2: f'Error loading domain metadata JSON: {meta_msg}',
        3: f'Error loading Grid .pkl: {grid_msg} & metadata JSON: {meta_msg}'
    }[status]

    # schema guard
    if meta:
        expected = geo_core.GridLayout.SCHEMA_ID
        found = meta.get('schema_id', None)
        if found != expected:
            status = 4
            msg = f'Unsupported schema: {found}; expected {expected}.'

    # otherwise return class object via class method
    if status:
        return status, msg, None
    payload: geo_core.GridPayload = {
        'schema_id': meta['schema_id'],
        'artifact_meta': meta['artifact_meta'],
        'data': tiles
    }
    return status, msg, geo_core.GridLayout.from_payload(payload)
