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
Canonical data blocks catalog.

This module defines a structured, dictionary-like container for managing
metadata entries associated with spatial data blocks. Each `CatalogEntry`
stores information such as file paths, spatial indices, validation stats,
and provenance metadata (e.g., hashes and timestamps).

The main class, BlocksCatalog, provides a mapping interface keyed by
(x, y) block coordinates, along with utilities for serialization to and
from JSON with deterministic ordering and formatting.
'''

# standard imports
from __future__ import annotations
import collections.abc
import json
import typing
# local imports
import landseg.geopipe.core as core
import landseg.utils as utils

# ---------------------------------Public Type---------------------------------
class CatalogEntry(typing.TypedDict):
    '''Typed dictionary representing metadata for a single data block.'''
    block_name: str
    file_path: str
    row_col: list[int]
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

class CatalogMeta(typing.TypedDict):
    '''Typed metadata dictionary for the current catalog.'''
    dataset: _DatasetInfo
    io_conventions: _IOConventions
    tensor_shapes: _TensorShapes
    labels: _LabelsInfo

# --------------------------------private  type--------------------------------
# ----- dataset
class _DatasetInfo(typing.TypedDict):
    '''Dataset identity and provenance metadata.'''
    name: str
    last_updated: str
    dataprep_commit: str
    mapped_grids: list[str]
    data_source: _DataSource

class _DataSource(typing.TypedDict):
    '''Input paths for fit/test images and labels.'''
    image_paths: list[str]
    label_paths: list[str]

# ----- io_conventions
class _IOConventions(typing.TypedDict):
    '''Block IO format, key names, shapes, dtypes, and ignore index.'''
    block_format: str
    shapes: _IOShapes
    dtypes: _IODtypes
    ignore_index: int

class _IOShapes(typing.TypedDict):
    '''Logical ordering conventions for tensors.'''
    image_order: str  # e.g., 'C,H,W'
    label_order: str  # e.g., 'L,H,W'

class _IODtypes(typing.TypedDict):
    '''Dtypes for serialized arrays.'''
    image: str
    label: str

# ----- tensor_shapes
class _TensorShapes(typing.TypedDict):
    '''Tensor shapes for image and label.'''
    image: _ImageTensorSpec
    label: _LabelTensorSpec

class _ImageTensorSpec(typing.TypedDict):
    '''Image tensor shape spec.'''
    order: str
    shape: list[int]  # [C, H, W]
    C: int
    H: int
    W: int

class _LabelTensorSpec(typing.TypedDict):
    '''Label tensor shape spec.'''
    order: str
    shape: list[int]  # [L, H, W]
    L: int
    H: int
    W: int

# ----- labels
class _LabelsInfo(typing.TypedDict):
    '''Labeling metadata.'''
    label_num_classes: int
    label_to_ignore: list[int]
    channel_parent: dict[str, str | None]
    channel_parent_cls: dict[str, int | None]

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

    def save_json(self, fpath: str) -> None:
        '''
        Save to JSON with deterministic ordering and custom formatting.

        The output is sorted by block coordinate keys and formatted with
        consistent indentation for readability. After writing, the file
        is hashed for integrity tracking.

        Args:
            fpath: Destination file path for the JSON output.
        '''

        # index entries and sort
        payload = {core.yx_name(k): v for k, v in self._data.items()}
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
        '''
        Create a `BlocksCatalog` instance from a JSON file.

        If the file does not exist, an empty catalog is returned.

        Args:
            fpath: Path to the JSON file to load.

        Returns:
            A `BlocksCatalog` instance populated with the file contents.
        '''

        obj = cls.__new__(cls)
        try:
            payload: dict[str, CatalogEntry] = utils.load_json(fpath)
        except FileNotFoundError:
            obj._data = {}
            return obj
        obj._data = {core.name_yx(k): v for k, v in payload.items()}
        return obj
