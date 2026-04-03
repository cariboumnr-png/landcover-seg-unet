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
NPZ related I/O helpers.
'''

# standard imports
import hashlib
import json
import os
import typing
import zipfile
import zlib
# third-party imports
import numpy

# -------------------------------Public Function-------------------------------
def load_dict_npz_hash(fpath: str) -> tuple[int, str, typing.Any]:
    '''
    Helper to load a npz file.

    return status:
    * `0`: loaded with correct hash record
    * `1`: loaded with mismatching hash record
    * `2`: loaded with no/corrupted hash record
    * `3`: npz file not found
    * `4`: npz file found but probably corrupted
    * `999`: other errors

    Returns:
        status, message, and loaded dict from the npz object.
    '''

    # parse
    filename = os.path.basename(fpath)
    hash_path = os.path.join(os.path.dirname(fpath), '_hash.json')

    try:
        data = numpy.load(fpath)
        # assert 'key' in data and 'values' in data
        loaded = {tuple(k): v for k, v in zip(data['keys'], data['values'])}
        file_hash_value = _get_hash(fpath)
        try:
            hash_record = _load_json(hash_path)
            if file_hash_value != hash_record.get(filename):
                return 1, f'Mismatching hash with record: {hash_path}', loaded
            return 0, f'NPZ loaded with hash check: {fpath} ', loaded
        except FileNotFoundError:
            return 2, f'Hash record not found : {hash_path}', loaded

    except FileNotFoundError:
        return 3, f'NPZ file not found: {fpath}', None

    except (zipfile.error, zlib.error):
        return 4, f'NPZ file not properly loaded: {fpath}', None

    return 999, 'Other errors occur during loading', None

def write_dict_npz_hash(
    fpath: str,
    d: typing.Mapping[typing.Any, numpy.ndarray]
) -> None:
    '''Helper to write a npz file from a python dict.'''

    # early exit
    if not d:
        raise ValueError("Cannot save empty dict")

    # parse fpath
    root = os.path.dirname(fpath)   # dir path
    fname = os.path.basename(fpath) # file name

    # make sure parent directory exsits before writing
    if root: # skip if write to root, e.g., ./file.npz
        os.makedirs(root, exist_ok=True)

    keys_list = list(d.keys())
    values_list = list(d.values())
    # --- Validate keys ---
    try:
        keys = numpy.array(keys_list)
    except Exception as e:
        raise TypeError('Keys cannot be converted to array') from e
    if keys.dtype == object:
        raise TypeError('Keys not stackable (ragged/inconsistent types/lens)')
    # --- Validate values ---
    try:
        values = numpy.stack(values_list)
    except Exception as e:
        raise ValueError(f'Values are not stackable: {e}') from e

    # write src to npz
    numpy.savez_compressed(fpath, keys=keys, values=values)

    # get src file hash
    hash_value = _get_hash(fpath)

    # load npz with some error handling (create new if so)
    hash_record_path = f'{root}/_hash.json'
    try:
        records = _load_json(hash_record_path)
    except (FileNotFoundError, json.JSONDecodeError):
        _write_json(hash_record_path, {'root': root})
        records = _load_json(hash_record_path)

    # append/update hash in record and save
    records[fname] = hash_value
    _write_json(hash_record_path, records)

# pure helpers
def _write_json(fp, src):
    with open(fp, 'w', encoding='UTF-8') as file:
        json.dump(src, file, indent=4)

def _load_json(fp):
    with open(fp, 'r', encoding='UTF-8') as src:
        return json.load(src)

def _get_hash(fp):
    with open(fp, 'rb') as file:
        sha256 = hashlib.sha256()
        for chunk in iter(lambda: file.read(8192), b''):
            sha256.update(chunk)
    return sha256.hexdigest()
