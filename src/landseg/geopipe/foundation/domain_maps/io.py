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

# standard imports
import os
# local imports
import landseg.geopipe.core as core
import landseg.utils as utils

# -------------------------------Public Function-------------------------------
def save_domain(
    domain_name: str,
    domain_obj: core.DomainTileMap,
    dirpath: str
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

    # prepare output dir
    os.makedirs(dirpath, exist_ok=True)

    # get domain object payload and pickle
    payload = domain_obj.to_payload()
    utils.write_json(f'{dirpath}/{domain_name}.json', payload)

    # write meta to json
    meta = {
        'schema_id': getattr(domain_obj, 'SCHEMA_ID', 'unknown'),
        'sha256': utils.hash_payload(payload),
        'grid_tile_size': payload['blk_size'],
        'mapped_tile_overlaps': payload['blk_overlap'],
        'index_range': payload['idx_range'],
        'context': payload['context'],
        'n_valid_tiles': len(domain_obj),
    }
    utils.write_json(f'{dirpath}/{domain_name}_meta.json', meta)

def load_domain(
    domain_name: str,
    dirpath: str,
    logger: utils.Logger
) -> core.DomainTileMap:
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
    payload = utils.load_json(f'{dirpath}/{domain_name}.json')
    meta = utils.load_json(f'{dirpath}/{domain_name}_meta.json')

    # schema guard
    expected = core.DomainTileMap.SCHEMA_ID
    found = meta.get('schema_id', None)
    if found != expected:
        raise ValueError(f'Unsupported schema: {found}; expected {expected}.')

    # otherwise return class object via class method
    return core.DomainTileMap.from_payload(payload, logger)
