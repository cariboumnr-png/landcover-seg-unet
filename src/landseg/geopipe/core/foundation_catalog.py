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
Canonical data blocks catalog and metadata typing utilities.

This module defines the core structures used to represent and manage a
catalog of spatial data blocks produced by the data-preparation pipeline.
It provides:

- `CatalogEntry`: a typed dictionary describing per-block metadata such
  as file paths, spatial indices, statistics, provenance hashes, and
  timestamps.

- `CatalogMeta` and related private TypedDicts: a structured schema
  capturing dataset-level metadata, I/O conventions, tensor shapes, and
  label specifications. These are used for describing the dataset as a
  whole and for generating or validating `metadata.json` files.

- `BlocksCatalog`: a dictionary-like container mapping (row, col) block
  coordinates to `CatalogEntry` objects. The class supports deterministic
  JSON serialization, reconstruction from disk, and standard mapping
  operations while permitting controlled mutation.

Together, these definitions provide a canonical representation for both
block-level and dataset-level metadata, ensuring consistency, type safety,
and reproducible serialization across the data pipeline.
'''

# standard imports
from __future__ import annotations
import collections.abc
import json
import typing
# local imports
import landseg.geopipe.utils as geo_utils

# ---------------------------------Public Type---------------------------------
class CatalogEntry(typing.TypedDict):
    '''Typed dictionary representing metadata for a single data block.'''
    block_name: str
    file_path: str
    row_col: list[int]
    base_valid_px: float
    base_class_count: list[int]
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
    '''
    A mapping container for managing catalog entries of spatial blocks.

    This class behaves like a read-only mapping from (row, col) coordinate
    tuples to `CatalogEntry` metadata dictionaries, while also supporting
    mutation via item assignment.

    Features:
    - Stores block metadata indexed by (x, y) coordinates
    - Provides dictionary-like access and iteration
    - Exposes utility methods for retrieving indexed file paths
    - Supports deterministic JSON serialization with custom formatting
    - Supports reconstruction from JSON files

    Notes:
        The internal storage is a standard Python dictionary. While the
        class implements the Mapping interface, it allows mutation
        via `__setitem__`.
    '''

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

    @classmethod
    def from_dict(cls, source: dict[str, CatalogEntry]) -> BlocksCatalog:
        '''
        Create a `BlocksCatalog` instance from a JSON file.

        If the file does not exist, an empty catalog is returned.

        Args:
            fpath: Path to the JSON file to load.

        Returns:
            A `BlocksCatalog` instance populated with the file contents.
        '''

        obj = cls.__new__(cls)
        obj._data = {geo_utils.name_xy(k): v for k, v in source.items()}
        return obj

    def to_json_payload(self) -> str:
        '''
        Deterministically ordered and custom-formatted JSON string.

        Returns:
            A formatted JSON string.
        '''

        # sort self._data and start the line
        payload = {geo_utils.xy_name(k): v for k, v in self._data.items()}
        items = sorted(payload.items())
        lines = ['{']
        # manual formatting line-by-line
        for i, (key, value) in enumerate(items):
            blk_txt = json.dumps(value)
            blk_txt = blk_txt.replace('{', '{\n\t\t')
            blk_txt = blk_txt.replace(', "', ',\n\t\t"')
            blk_txt = blk_txt.replace('}', '\n\t}')

            line = f'\t"{key}": {blk_txt}'
            if i < len(items) - 1:
                line += ','
            lines.append(line)

        lines.append('}')
        return '\n'.join(lines)
