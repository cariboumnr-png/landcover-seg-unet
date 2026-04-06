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
import typing
# local imports
import landseg.artifacts as artifacts

class PayloadDict(typing.TypedDict):
    '''
    Generic serialization payload dictionary.
    '''
    schema_id: str
    artifact_meta: typing.Mapping[str, typing.Any]
    data: typing.Mapping[str, typing.Any]

# -------------------------------Public Function-------------------------------
def save_payload(
    payload: object,
    name: str,
    dirpath: str,
) -> None:
    '''
    Serialize a payload to disk.
    '''

    # basic validation
    if not isinstance(payload, dict):
        raise TypeError("payload must be a dict")

    required = {'schema_id', 'artifact_meta', 'data'}
    missing = required - payload.keys()
    if missing:
        raise ValueError(f"Missing payload keys: {missing}")
    # type casting
    payload = typing.cast(PayloadDict, payload)

    # get data (pop and rest goes to meta)
    payload = dict(payload)  # shallow copy
    data = payload.pop('data')

    # payload and meta path
    data_path = f'{dirpath}/{name}.json'
    meta_path = f'{dirpath}/{name}_meta.json'

    policy = artifacts.LifecyclePolicy.BUILD_IF_MISSING # dummy

    # artifact controllers
    data_ctrl = artifacts.Controller(data_path, 'json', policy)
    meta_ctrl = artifacts.Controller(meta_path, 'json', policy)

    # write domain tiles dict and write to JSON
    data_ctrl.persist(data)
    # write meta dict and write to json
    meta_ctrl.persist(payload)

def load_payload(
    name: str,
    dirpath: str,
    schema_id: str,
    policy: artifacts.LifecyclePolicy
) -> PayloadDict | None:
    '''
    doc
    '''

    # payload and meta path
    data_path = f'{dirpath}/{name}.json'
    meta_path = f'{dirpath}/{name}_meta.json'

    # artifact controllers
    data_ctrl = artifacts.Controller[dict[str, typing.Any]](data_path, 'json', policy)
    meta_ctrl = artifacts.Controller[dict[str, typing.Any]](meta_path, 'json', policy)

    # fetch
    try:
        data = data_ctrl.fetch()
    except artifacts.ArtifactError as exc:
        raise artifacts.ArtifactError(
            f'Error loading {data_path}: {exc}'
        ) from exc
    try:
        meta = meta_ctrl.fetch()
    except artifacts.ArtifactError as exc:
        raise artifacts.ArtifactError(
            f'Error loading {meta_path}: {exc}'
        ) from exc

    # loading status
    if meta is None or data is None:
        return None

    # schema guard
    found = meta.get('schema_id', None)
    if found != schema_id:
        raise artifacts.ArtifactError(
            f'Unsupported schema: {found}; expected {schema_id}.'
        )

    # otherwise return class object via class method
    payload: PayloadDict = {
        'schema_id': meta['schema_id'],
        'artifact_meta': meta['artifact_meta'],
        'data': data
    }
    return payload
