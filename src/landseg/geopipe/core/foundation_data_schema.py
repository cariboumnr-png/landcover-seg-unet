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
Dataset-level metadata schema for block catalogs.

This module defines structured TypedDict schemas used to describe
dataset-wide metadata for a catalog of spatial data blocks. It captures
information about dataset identity, data sources, I/O conventions, tensor
shapes, and label specifications.

These schemas provide a consistent contract for generating, validating,
and consuming `metadata.json` files, ensuring interoperability and
reproducibility across the data pipeline.
'''

# standard imports
from __future__ import annotations
import typing

SCHEMA_ID = 'data_schema/v1'

# ---------------------------------Public Type---------------------------------
class DataSchema(typing.TypedDict):
    '''
    Top-level metadata structure for a block catalog.

    This schema aggregates all dataset-level metadata required to
    interpret stored blocks, including provenance, I/O conventions,
    tensor layouts, and labeling configuration.

    Fields:

    - **dataset**:
        Dataset identity and provenance information.
    - **io_conventions**:
        Serialization format, tensor ordering, and dtype rules.
    - **tensor_shapes**:
        Explicit tensor shape specifications for image and label data.
    - **labels**:
        Label schema, hierarchy, and ignore rules.

    **Schema**: `SCHEMA_ID` = `'blocks_catalog_payload/v1'`
    '''

    schema_id: str
    dataset: _DatasetInfo
    io_conventions: _IOConventions
    tensor_shapes: _TensorShapes
    labels: _LabelsInfo

# --------------------------------private  type--------------------------------
# ----- dataset
class _DatasetInfo(typing.TypedDict):
    '''Dataset identity and provenance information.'''
    name: str
    last_updated: str
    dataprep_commit: str
    mapped_grids: list[str]
    data_source: _DataSource

class _DataSource(typing.TypedDict):
    '''Input data source paths for images and labels.'''
    image_paths: list[str]
    label_paths: list[str]

# ----- io_conventions
class _IOConventions(typing.TypedDict):
    '''Serialization format, tensor ordering, and ignore index rules.'''
    block_format: str
    shapes: _IOShapes
    dtypes: _IODtypes
    ignore_index: int

class _IOShapes(typing.TypedDict):
    '''Logical dimension ordering for image and label tensors.'''
    image_order: str  # e.g., 'C,H,W'
    label_order: str  # e.g., 'L,H,W'

class _IODtypes(typing.TypedDict):
    '''Data types used for serialized image and label arrays.'''
    image: str
    label: str

# ----- tensor_shapes
class _TensorShapes(typing.TypedDict):
    '''Container for image and label tensor shape specifications.'''
    image: _ImageTensorSpec
    label: _LabelTensorSpec

class _ImageTensorSpec(typing.TypedDict):
    '''Shape and dimension metadata for image tensors.'''
    order: str
    shape: list[int]  # [C, H, W]
    C: int
    H: int
    W: int

class _LabelTensorSpec(typing.TypedDict):
    '''Shape and dimension metadata for label tensors.'''
    order: str
    shape: list[int]  # [L, H, W]
    L: int
    H: int
    W: int

# ----- labels
class _LabelsInfo(typing.TypedDict):
    ''''Label schema, hierarchy, and ignore configuration.'''
    label_num_classes: int
    label_to_ignore: list[int]
    channel_parent: dict[str, str | None]
    channel_parent_cls: dict[str, int | None]
