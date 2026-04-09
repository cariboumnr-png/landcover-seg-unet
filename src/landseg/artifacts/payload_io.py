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
I/O utilities for split-file-style artifacts.

This module provides a typed controller for managing a pair of JSON
artifacts representing a logical payload split across two files:

- a *data* file (e.g., `x.json`) containing the primary serialized object
- a *metadata* sidecar (e.g., `x_meta.json`) containing schema and
  auxiliary information

The controller enforces:
- schema validation via a `schema_id`
- coordinated loading/saving of both files
- consistent error handling through the artifacts subsystem

A payload is represented as a dictionary with three fields:
    - `schema_id`: identifier for compatibility validation
    - `artifact_meta`: arbitrary metadata associated with the payload
    - `data`: the primary serialized content
'''

# standard imports
import os
import typing
# local imports
import landseg.artifacts as artifacts

# TypeVars
D = typing.TypeVar('D') # 'data' payload, e.g., x.json
M = typing.TypeVar('M') # 'meta' payload, e.g., x_meta.json

# -----------------------------private type-----------------------------
class _PayloadDict(typing.TypedDict, typing.Generic[D, M]):
    '''
    Strongly-typed representation of a split-file payload.

    This structure combines the contents of the data file and its
    corresponding metadata sidecar into a single logical object.

    Type Parameters:
        D: Type of the primary data payload (data file contents)
        M: Type of the metadata payload (artifact_meta field)

    Fields:
        schema_id:
            Identifier used to validate compatibility between stored
            artifacts and the expected schema.
        artifact_meta:
            Arbitrary metadata associated with the artifact, stored in
            the sidecar file.
        data:
            The main serialized payload loaded from the data file.
    '''
    schema_id: str
    artifact_meta: M
    data: D

# -----------------------------Public Class-----------------------------
class PayloadController(typing.Generic[D, M]):
    '''
    Coordinates persistence and retrieval of a split JSON payload.

    This controller manages two underlying artifact files:
        1. A data file containing the primary serialized payload
        2. A metadata sidecar containing schema and auxiliary metadata

    It ensures:
        - both files are loaded together
        - schema compatibility is enforced on load
        - consistent persistence using the configured lifecycle policy

    Type Parameters:
        D: Type of the primary data payload
        M: Type of the metadata payload
    '''

    def __init__(
        self,
        data_fpath: str,
        *,
        schema_id: str,
        policy: artifacts.LifecyclePolicy
    ):
        '''
        Initialize a controller for a split-file payload.

        Args:
            data_fpath:
                Path to the primary data JSON file. The metadata file
                path is derived automatically by appending `_meta.json`
                to the base filename.
            schema_id:
                Expected schema identifier used to validate payloads
                during loading.
            policy:
                Lifecycle policy governing how artifacts are read and
                written (e.g., overwrite rules, caching, etc.).
        '''

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
        '''
        Load and validate the split-file payload from disk.

        Behavior
        - Loads data and metadata independently
        - Verifies schema compatibility using `schema_id`
        - Merges both into a single structured payload on success

        Returns:
            A `_PayloadDict` containing `schema_id`, `artifact_meta`,
            and `data` if both files are successfully loaded and valid.

            or `None` if either the data or metadata artifact is
            missing (as indicated by the underlying controllers).

        Raises:
            artifacts.ArtifactError:
                If an error occurs while fetching either artifact, or
                If the stored `schema_id` does not match the expected
                schema for this controller
        '''

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
        Persist a split-file payload to disk.

        Behavior:
            - Validates payload structure
            - Writes data and metadata independently using their
              respective artifact controllers
            - Does not re-validate `schema_id` consistency; assumes
              caller provides a correct payload

        The payload is written as two separate JSON artifacts:
            - `data` → data file
            - `{schema_id, artifact_meta}` → metadata sidecar

        Args:
            payload:
                A dictionary containing:
                    - `schema_id`
                    - `artifact_meta`
                    - `data`

        Raises:
            TypeError:
                If `payload` is not a dictionary.
            ValueError:
                If required keys are missing from the payload.
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
