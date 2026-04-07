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
General utility functions used throughout this project.
'''

# standard imports
import datetime
import hashlib
import json
import os
import sys
import typing

# -------------------------------Public Function-------------------------------
def get_file_ctime(filepath: str, t_format: str='%Y%m%d_%H%M%S') -> str:
    '''
    Get file creation time as a string specified by `t_format`.

    Args:
        filepath (str): To the file to be checked.
        t_format (str, optional): Sets time string format
            (default: 20001234_567).
    '''

    # get creation time
    creation_time = os.path.getctime(filepath)
    # format and return
    return datetime.datetime.fromtimestamp(creation_time).strftime(t_format)

def load_json(json_fpath: str) -> typing.Any:
    '''Generic helper to load a json config file.'''

    with open(json_fpath, 'r', encoding='UTF-8') as src:
        return json.load(src)

def write_json(json_fpath: str, src_dict: list | dict | typing.Mapping) -> None:
    '''Generic helper to write a JSON from a python dict or list.'''

    # make sure parent directory exsits before writing
    dirpath = os.path.dirname(json_fpath)
    if dirpath: # skip if write to root, e.g., ./file.json
        os.makedirs(dirpath, exist_ok=True)
    with open(json_fpath, 'w', encoding='UTF-8') as file:
        json.dump(src_dict, file, indent=4)

def hash_sha256(fpath: str) -> str:
    '''Hash an artifact and write to a hash file.'''

    # get hash as a string and return
    with open(fpath, 'rb') as file:
        sha256 = hashlib.sha256()
        for chunk in iter(lambda: file.read(8192), b''):
            sha256.update(chunk)
    hash_value = sha256.hexdigest()
    return hash_value

def print_status(lines: list):
    '''Helper to print multiple lines refreshing'''

    print('\n')
    # Calculate the number of lines that should be refreshed
    num_lines_to_clear = len(lines)
    # Move the cursor up by that number of lines
    sys.stdout.write(f'\033[{num_lines_to_clear}F')
    # Move cursor up by len(lines) and clear lines using ANSI escape codes
    sys.stdout.write('\033[F' * len(lines))  # Move cursor up
    for line in lines:
        sys.stdout.write('\033[K')  # Clear the line
        print(line)
    print('\n')
