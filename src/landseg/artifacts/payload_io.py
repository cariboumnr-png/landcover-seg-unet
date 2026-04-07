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
import typing
# local imports
import landseg.artifacts as artifacts

# TypeVars
D = typing.TypeVar('D') # 'data' payload, e.g., x.json
M = typing.TypeVar('M') # 'meta' payload, e.g., x_meta.json

class _PayloadDict(typing.TypedDict, typing.Generic[D, M]):
    '''A strictly typed container for a split-file artifact.'''
    schema_id: str
    artifact_meta: M
    data: D

class PayloadController(typing.Generic[D, M]):
    '''
    Manages a pair of artifacts (data + meta) with strict typing.
    '''

    def __init__(
        self,
        data_fpath: str,
        schema_id: str,
        policy: artifacts.LifecyclePolicy
    ):
        '''doc'''

        # init attrs
        self.schema_id = schema_id
        self.policy = policy

        # compile file paths
        no_ext, _ = os.path.splitext(data_fpath)
        self.data_path = data_fpath
        self.meta_path = f'{no_ext}_meta.json'

        # controllers
        self.data_ctrl = artifacts.Controller(self.data_path, policy)
        self.meta_ctrl = artifacts.Controller(self.meta_path, policy)

    def load(self) -> _PayloadDict[D, M] | None:
        '''doc'''

        # fetch
        try:
            data = self.data_ctrl.fetch()
        except artifacts.ArtifactError as exc:
            raise artifacts.ArtifactError(
                f'Error loading {self.data_path}: {exc}'
            ) from exc
        try:
            meta = self.meta_ctrl.fetch()
        except artifacts.ArtifactError as exc:
            raise artifacts.ArtifactError(
                f'Error loading {self.meta_path}: {exc}'
            ) from exc

        # loading status
        if meta is None or data is None:
            return None

        # schema guard
        found = meta.get('schema_id', None)
        if found != self.schema_id:
            _ = f'Mismatch schema: {found}; expected {self.schema_id}.'
            raise artifacts.ArtifactError(_)

        # otherwise return class object via class method
        payload: _PayloadDict[D, M] = {
            'schema_id': self.schema_id,
            'artifact_meta': meta.get('artifact_meta'),
            'data': data
        }
        return payload

    def save(self, payload: _PayloadDict[D, M]) -> None:
        '''
        Serialize the split-file payload to disk.
        '''

        # basic validation
        if not isinstance(payload, dict):
            raise TypeError("payload must be a dict")

        required = {'schema_id', 'artifact_meta', 'data'}
        missing = required - payload.keys()
        if missing:
            raise ValueError(f"Missing payload keys: {missing}")

        # get data and meta
        data = payload['data']
        meta = {k: payload[k] for k in ['schema_id','artifact_meta']}

        # write domain tiles dict and write to JSON
        self.data_ctrl.persist(data)
        # write meta dict and write to json
        self.meta_ctrl.persist(meta)
