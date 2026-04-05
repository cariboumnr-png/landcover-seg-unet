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
Artifact lifecycle controller.

This module provides a generic controller for managing persisted
artifacts with lifecycle policies, integrity validation via SHA256,
and support for JSON and NumPy-based storage formats.
'''

# standard imports
from __future__ import annotations
import hashlib
import json
import os
import typing
import zipfile
import zlib
# third-party imports
import numpy
# local imports
import landseg.artifacts as artifacts

T = typing.TypeVar('T')

class ArtifactController(typing.Generic[T]):
    '''
    Manage loading, validation, and persistence of disk-based artifacts.

    The controller enforces lifecycle policies, tracks file integrity
    via SHA256 hashing, and supports multiple serialization formats.
    '''

    def __init__(
        self,
        file_path: str,
        file_type: typing.Literal['json', 'npz_dict'],
        policy: artifacts.LifecyclePolicy,
    ):
        '''
        Initialize the controller for a single artifact.

        Args:
            file_path: Absolute or relative path to the artifact file.
            file_type: Serialization format used for persistence.
            policy: Lifecycle policy governing load and rebuild behavior.
        '''

        self.fp = file_path
        self.dir = os.path.dirname(self.fp)
        self.fname = os.path.basename(file_path)
        self.hash_fpath = os.path.join(self.dir, '_hash.json')
        self.type = file_type
        self.policy = policy

    @property
    def is_valid(self) -> bool:
        '''Return True if the artifact exists and its hash is valid.'''
        return os.path.exists(self.fp) and self._check_sha256()

    def fetch(self) -> T | None:
        '''
        Fetch the artifact according to the lifecycle policy.

        Returns:
            The loaded artifact if valid and permitted by the policy,
            otherwise None to signal that reconstruction is required.

        Raises:
            ArtifactError: If the artifact exists but is corrupted or
                fails integrity validation.
        '''

        data: T | None
        try:
            data = self._load()
        except ArtifactError as exc:
            raise ArtifactError from exc

        match self.policy:
            # policy: build if missing
            case artifacts.LifecyclePolicy.BUILD_IF_MISSING:
                return data
            # policy: force rebuild
            case artifacts.LifecyclePolicy.REBUILD:
                return None
            # unsupported policy
            case _:
                raise NotImplementedError(f'Unsupported policy: {self.policy}')

    def persist(self, src: typing.Any):
        '''
        Persist an artifact to disk and update its hash record.

        Args:
            src: Serializable artifact object to be written.

        Raises:
            ValueError: If the input data is invalid for the chosen
                serialization format.
        '''

        # make sure parent directory exsits before writing
        if self.dir: # skip if write to root, e.g., ./file.json
            os.makedirs(self.dir, exist_ok=True)

        # write files according to type
        match self.type:
            case 'json':
                self._json_write(self.fp, src)
            case 'npz_dict':
                self._npz_write_dict(self.fp, src)

        # get hash after file is written
        sha256 = self._get_sha256(self.fp)

        # load hash record with some error handling (create new if so)
        try:
            records = self._json_read(self.hash_fpath)
        except (FileNotFoundError, json.JSONDecodeError):
            self._json_write(self.hash_fpath, {'root': self.dir})
            records = self._json_read(self.hash_fpath)

        # append/update hash in record and save
        records[self.fname] = sha256
        self._json_write(self.hash_fpath, records)

    def _load(self) -> typing.Any:
        '''Load an artifact and verify its integrity.'''

        try:
            match self.type:
                case 'json': loaded = self._json_read(self.fp)
                case 'npz_dict': loaded = self._npz_read_dict(self.fp)
                case _: raise ValueError(f'Unsupported type: {self.type}')
            if self._check_sha256():
                return loaded
            raise _ArtifactHashMismatch
        except FileNotFoundError as exc:
            raise _ArtifactMissing from exc
        except (json.JSONDecodeError, zipfile.error, zlib.error) as exc:
            raise _ArtifactCorrupted from exc

    def _check_sha256(self) -> bool:
        '''Check artifact hash against the recorded value.'''

        sha256_value = self._get_sha256(self.fp)
        try:
            sha256_record = self._json_read(self.hash_fpath).get(self.fname)
            return sha256_value == sha256_record
        except (FileNotFoundError, json.JSONDecodeError):
            return False

    @staticmethod
    def _get_sha256(fp):
        '''Compute SHA256 checksum of a file.'''
        with open(fp, 'rb') as file:
            sha256 = hashlib.sha256()
            for chunk in iter(lambda: file.read(8192), b''):
                sha256.update(chunk)
        return sha256.hexdigest()

    @staticmethod
    def _json_read(fp):
        '''Read JSON data from disk.'''

        with open(fp, 'r', encoding='UTF-8') as src:
            return json.load(src)

    @staticmethod
    def _json_write(fp, src):
        '''Write JSON data or formatted JSON string to disk.'''

        if isinstance(src, str): # special case: formatted JSON string
            with open(fp, 'w', encoding='UTF-8') as file:
                file.write(src)
        else:
            with open(fp, 'w', encoding='UTF-8') as file:
                json.dump(src, file, indent=4)

    @staticmethod
    def _npz_read_dict(fp):
        '''Load a dictionary of arrays from a compressed NPZ file.'''
        data = numpy.load(fp)
        return dict(zip(data['keys'], data['values']))

    @staticmethod
    def _npz_write_dict(fp, src):
        '''Persist a dictionary of NumPy arrays as a compressed NPZ.'''

        # early exits
        if not src:
            raise ValueError('Cannot save empty dict')
        if not (
            isinstance(src, dict) and
            all(isinstance(v, numpy.ndarray) for v in src.values())
        ):
            raise ValueError('Input source must be a dictionary of arrays')

        # get keys and values
        keys_list = list(src.keys())
        values_list = list(src.values())

        # validate keys
        try:
            keys = numpy.array(keys_list)
        except Exception as e:
            raise TypeError('Keys cannot be converted to array') from e
        if keys.dtype == object:
            raise TypeError('Keys not stackable (ragged/inconsistent types/lens)')
        # validate values
        try:
            values = numpy.stack(values_list)
        except Exception as e:
            raise ValueError(f'Values are not stackable: {e}') from e

        # write src to npz
        numpy.savez_compressed(fp, keys=keys, values=values)

class ArtifactError(Exception):
    '''Base class for artifact-related errors.'''
    def __init__(self, message='Error loading artifact'):
        super().__init__(message)

class _ArtifactCorrupted(ArtifactError):
    def __init__(self, message='Artifact file probably corrupted'):
        super().__init__(message)

class _ArtifactHashMismatch(ArtifactError):
    def __init__(self, message='Artifact file hash value mismatch'):
        super().__init__(message)

class _ArtifactMissing(ArtifactError):
    def __init__(self, message='Artifact file not found on disk'):
        super().__init__(message)
