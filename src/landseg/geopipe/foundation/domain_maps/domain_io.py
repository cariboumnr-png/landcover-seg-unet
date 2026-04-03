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
I/O utilites for class `DomainTileMap`.

This module handles persistence of `DomainTileMap` objects via a JSON
payload and a JSON metadata sidecar.

A schema identifier and a SHA256 hash are used to validate payload
compatibility and integrity on load.
'''

# local imports
import landseg.geopipe.artifacts as artifacts
import landseg.geopipe.core as geo_core

# -------------------------------Public Function-------------------------------
def save_domain(
    obj: geo_core.DomainTileMap,
    name: str,
    dirpath: str,
) -> None:
    '''
    Serialize a `DomainTileMap` to disk.

    Writes a JSON payload and a JSON metadata file containing schema
    identifier, integrity hash, and summary fields.

    Args:
        grid_name: Name of the grid associated with this domain.
        domain_name: Name of the domain in both JSON artifacts.
        domain_obj: The `DomainTileMap` instance to be saved.
        dirpath: Directory where artifacts are to be saved.
    '''

    payload = obj.to_json_payload()
    # write domain tiles dict and write to JSON
    artifacts.write_json_hash(f'{dirpath}/{name}.json', payload['tiles_dict'])
    # write meta dict and write to json
    artifacts.write_json_hash(f'{dirpath}/{name}_meta.json',payload['meta'])

def load_domain(
    name: str,
    dirpath: str,
) -> tuple[int, str, geo_core.DomainTileMap | None]:
    '''
    Load a `DomainTileMap` from disk.

    Validates payload hash and schema identifier before reconstructing
    the `DomainTileMap` instance. Raises if the payload is corrupted or the
    schema is unsupported.

    Args:
        domain_name: Name of the domain in both JSON artifacts.
        dirpath: Directory where artifacts are located.
    '''

    # load payload and meta json
    payload_path = f'{dirpath}/{name}.json'
    meta_path = f'{dirpath}/{name}_meta.json'
    # types declaration
    tiles: dict[str, geo_core.DomainTile]
    meta: geo_core.DomainMetadata
    tiles_status, tiles_msg, tiles = artifacts.load_json_hash(payload_path)
    meta_status, meta_msg, meta = artifacts.load_json_hash(meta_path)

    # loading status
    status = {
        (False, False): 0,
        (True, False): 1,
        (False, True): 2,
        (True, True): 3
    }[bool(tiles_status), bool(meta_status)]

    # combined summary message
    msg = {
        0: 'Domain JSON and domain metadata JSON loaded successfully',
        1: f'Error loading domain JSON: {tiles_msg}',
        2: f'Error loading domain metadata JSON: {meta_msg}',
        3: f'Error loading domain JSON: {tiles_msg} & metadata JSON: {meta_msg}'
    }[status]

    # schema guard
    if meta:
        expected = geo_core.DomainTileMap.SCHEMA_ID
        found = meta.get('schema_id', None)
        if found != expected:
            status = 4
            msg = f'Unsupported schema: {found}; expected {expected}.'

    # otherwise return class object via class method
    if status:
        return status, msg, None
    payload: geo_core.DomainPayload = {'meta': meta, 'tiles_dict': tiles}
    return status, msg, geo_core.DomainTileMap.from_json_payload(payload)
