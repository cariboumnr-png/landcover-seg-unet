'''
General utility functions used throughout this project.
'''

# standard imports
import datetime
import hashlib
import json
import os
import pathlib
import pickle
import sys
import typing

# -------------------------------Public Function-------------------------------
def get_dir_size(dirpath: str, pattern: str | None=None) -> int:
    '''Get total size of files (optionally filtered by glob type).'''

    pattern = pattern or '*'
    return sum(
        f.stat().st_size
        for f in pathlib.Path(dirpath).rglob(pattern) if f.is_file()
    )

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

def get_timestamp(t_format: str='%Y%m%d_%H%M%S') -> str:
    '''
    Get current time as a string specified by `t_format`.

    Args:
        t_format (str, optional): Sets time string format
            (default: 20001234_567).
    '''

    # return formatted time string
    return datetime.datetime.now().strftime(t_format)

def load_json(json_fpath: str) -> typing.Any:
    '''Helper to load a json config file.'''

    with open(json_fpath, 'r', encoding='UTF-8') as src:
        return json.load(src)

def load_pickle(pickle_fpath: str) -> typing.Any:
    '''Helper to load a .pickle file'''

    with open(pickle_fpath, 'rb') as file:
        return pickle.load(file)

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

def write_json(json_fpath: str, src_dict: list | dict | typing.Mapping) -> None:
    '''Helper to write a json config file from a python dict or list.'''

    # make sure parent directory exsits before writing
    dirpath = os.path.dirname(json_fpath)
    os.makedirs(dirpath, exist_ok=True)
    with open(json_fpath, 'w', encoding='UTF-8') as file:
        json.dump(src_dict, file, indent=4)

def write_pickle(pickle_fpath: str, src_obj: typing.Any) -> None:
    '''Helper to write a json config file from a python dict or list.'''

    # make sure parent directory exsits before writing
    dirpath = os.path.dirname(pickle_fpath)
    os.makedirs(dirpath, exist_ok=True)
    with open(pickle_fpath, 'wb') as file:
        pickle.dump(src_obj, file)

def hash_artifacts(fpath: str, write_to_record: bool = True) -> str:
    '''Hash an artifact and write to a hash file.'''

    # get hash as a string
    with open(fpath, 'rb') as file:
        sha256 = hashlib.sha256()
        for chunk in iter(lambda: file.read(8192), b''):
            sha256.update(chunk)
    hash_value = sha256.hexdigest()

    # if needs to write to a record at the root
    if write_to_record:
        # get root and name of the file
        root = os.path.dirname(fpath)
        fname = os.path.basename(fpath)
        # create a hash record file at the root if not already present
        hash_record_path = f'{root}/hash.json'
        if not os.path.exists(hash_record_path):
            write_json(hash_record_path, {'root': root})
        # check hash in record
        records = load_json(hash_record_path)
        records[fname] = hash_value
        # update record
        write_json(hash_record_path, records)

    # return
    return hash_value

def hash_payload(payload: typing.Any) -> str:
    '''
    Hash a JSON-serializable payload.

    Raises:
        TypeError if the payload is not JSON-serializable.
    '''

    blob = json.dumps(payload, sort_keys=True, separators=(',', ':'))
    sha256 = hashlib.sha256(blob.encode()).hexdigest()
    return sha256
