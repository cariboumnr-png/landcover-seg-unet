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
JSON related I/O helpers.
'''

# standard imports
import hashlib
import json
import os
import typing

# -------------------------------Public Function-------------------------------
def load_json_hash(fpath: str) -> tuple[int, str, typing.Any]:
    '''
    Helper to load a json config file.

    return status:
    * `0`: loaded with correct hash record
    * `1`: loaded with mismatching hash record
    * `2`: loaded with no/corrupted hash record
    * `3`: JSON file not found
    * `4`: Json file found but probably corrupted
    999: other errors

    Returns:
        status, message, and loaded JSON object
    '''

    # parse
    filename = os.path.basename(fpath)
    hash_path = os.path.join(os.path.dirname(fpath), '_hash.json')

    try:
        loaded = _load(fpath)
        file_hash_value = _get_hash(fpath)
        try:
            hash_record = _load(hash_path)
            if file_hash_value != hash_record.get(filename):
                return 1, f'JSON hash value not matching with record: {hash_path}', loaded
            return 0, f'JSON loaded with hash check successfully: {fpath} ', loaded
        except (FileNotFoundError, json.JSONDecodeError):
            return 2, f'Hash record not found or corrupted: {hash_path}', loaded

    except FileNotFoundError:
        return 3, f'JSON file not found: {fpath}', None

    except json.JSONDecodeError:
        return 4, f'JSON file not properly loaded: {fpath}', None

    return 999, 'Other errors occur during loading', None

def write_json_hash(
    fpath: str,
    src: list | dict | typing.Mapping | typing.Any
) -> None:
    '''Helper to write a json config file from a python dict or list.'''

    # parse fpath
    root = os.path.dirname(fpath)   # dir path
    fname = os.path.basename(fpath) # file name

    # make sure parent directory exsits before writing
    if root: # skip if write to root, e.g., ./file.json
        os.makedirs(root, exist_ok=True)

    # write src to json
    if isinstance(src, str): # special case: formatted string
        with open(fpath, 'w', encoding='UTF-8') as file:
            file.write(src)
    else:
        _write(fpath, src)

    # get src file hash
    hash_value = _get_hash(fpath)

    # load json with some error handling (create new if so)
    hash_record_path = f'{root}/_hash.json'
    try:
        records = _load(hash_record_path)
    except (FileNotFoundError, json.JSONDecodeError):
        _write(hash_record_path, {'root': root})
        records = _load(hash_record_path)

    # append/update hash in record and save
    records[fname] = hash_value
    _write(hash_record_path, records)

# pure helpers
def _write(fp, src):
    with open(fp, 'w', encoding='UTF-8') as file:
        json.dump(src, file, indent=4)

def _load(fp):
    with open(fp, 'r', encoding='UTF-8') as src:
        return json.load(src)

def _get_hash(fp):
    with open(fp, 'rb') as file:
        sha256 = hashlib.sha256()
        for chunk in iter(lambda: file.read(8192), b''):
            sha256.update(chunk)
    return sha256.hexdigest()
