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
doc
'''

# standard imports
from __future__ import annotations
import collections.abc
import json
import typing
# local imports
import landseg.utils as utils

# ---------------------------------Public Type---------------------------------
class CatalogEntry(typing.TypedDict):
    '''Catalog entry per block.'''
    block_name: str
    file_path: str
    loc_col_row: list[int]
    valid_px: float
    class_count: list[int]
    schema_version: str
    creation_time: str
    sha_256: str
    aligned_grid: str
    source_image: str
    source_image_sha_256: str
    source_label: str | None
    source_label_sha_256: str | None

# --------------------------------Public  Class--------------------------------
class BlocksCatalog(collections.abc.Mapping[tuple[int, int], CatalogEntry]):
    '''doc'''

    SCHEMA_ID: str = 'v_1.0'

    def __init__(self) -> None:
        self._data: dict[tuple[int, int], CatalogEntry] = {}

    def __getitem__(self, key: tuple[int, int]) -> CatalogEntry:
        return self._data[key]

    def __setitem__(self, key: tuple[int, int], value):
        self._data[key] = value

    def __iter__(self) -> collections.abc.Iterator[tuple[int, int]]:
        return iter(self._data)

    def __len__(self) -> int:
        return len(self._data)

    @property
    def indexed_block_files(self) -> dict[tuple[int, int], str]:
        '''Return the catalogued block file list.'''
        return {k: v['file_path'] for k, v in self._data.items()}

    def save_json(self, fpath: str) -> None:
        '''Save the catalog as a json with custom cosmetics.'''

        # index entries and sort
        payload = {_xy_name(k): v for k, v in self._data.items()}
        sorted_payload = dict(sorted(payload.items()))
        # manual json writing
        with open(fpath, 'w', encoding='UTF-8') as file:
            file.write('{\n')
            for i, (key, value) in enumerate(sorted_payload.items()):
                blk_txt = json.dumps(value)
                blk_txt = blk_txt.replace('{', '{\n\t\t')
                blk_txt = blk_txt.replace(', "', ',\n\t\t"')
                blk_txt = blk_txt.replace('}', '\n\t}')
                file.write(f'\t"{key}": {blk_txt}')
                if i == len(sorted_payload) - 1:
                    file.write('\n')
                else:
                    file.write(',\n')
            file.write('}\n')
        # hash
        utils.hash_artifacts(fpath)

    @classmethod
    def from_json(cls, fpath: str) -> BlocksCatalog:
        '''Load a .json file and instantiate a class instance.'''

        obj = cls.__new__(cls)
        try:
            payload: dict[str, CatalogEntry] = utils.load_json(fpath)
        except FileNotFoundError:
            obj._data = {}
            return obj
        obj._data = {_name_xy(k): v for k, v in payload.items()}
        return obj


# helpers
def _xy_name(coords: tuple[int, int]) -> str:
    '''
    Convert (x, y) coordinates to a canonical block name string:
    `(12, 34)` -> `'col_000012_row_000034'`.
    '''

    x, y = coords
    return f'col_{x:06d}_row_{y:06d}'

def _name_xy(name: str) -> tuple[int, int]:
    '''
    Convert a canonical block name back to coordinates:
    `'col_000012_row_000034'` -> `(12, 34)`.
    '''

    split = name.split('_')
    x_str, y_str = split[1], split[3]
    return int(x_str), int(y_str)
