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
`CatalogMeta` and related private TypedDicts: a structured schema
  capturing dataset-level metadata, I/O conventions, tensor shapes, and
  label specifications. These are used for describing the dataset as a
  whole and for generating or validating `metadata.json` files.
'''

# standard imports
from __future__ import annotations
import typing

# ---------------------------------Public Type---------------------------------
class BlocksMetadata(typing.TypedDict):
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
