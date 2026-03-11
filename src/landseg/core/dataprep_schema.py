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

'''`Landseg.dataprep` schema definitions.'''

# standard imports
from __future__ import annotations
import typing

# ---------------------------------Public Type---------------------------------
class SchemaOneBlock(typing.TypedDict):
    '''Minimal schema emitted by build_schema_from_a_block(...).
    '''
    dataset_name: str
    image_channel: int
    ignore_index: int
    block_size: int
    class_counts: dict[str, list[int]]
    logit_adjust: dict[str, list[float]]
    head_parent: dict[str, str | None]
    head_parent_cls: dict[str, int | None]
    train_split: dict[str, str]
    val_split: dict[str, str]

class SchemaFull(typing.TypedDict):
    '''Dataset-wide schema emitted by build_schema(...).'''
    schema_version: str
    dataset: _DatasetInfo
    world_grid: _WorldGridInfo
    io_conventions: _IOConventions
    tensor_shapes: _TensorShapes
    labels: _LabelsInfo
    splits: _SplitsInfo
    training_stats: _TrainingStatsInfo
    normalization: _NormalizationInfo
    checksums: _Checksums

# --------------------------------private  type--------------------------------
# ----- dataset
class _DatasetInfo(typing.TypedDict):
    '''Dataset identity and provenance metadata.'''
    name: str
    created_at: str
    dataprep_commit: str
    data_source: _DataSource
    has_test_data: bool
    test_data_has_label: bool

class _DataSource(typing.TypedDict):
    '''Input paths for fit/test images and labels.'''
    fit_image_path: str
    fit_label_path: str
    test_image_path: str | None
    test_label_path: str | None

# ----- world_grid
class _WorldGridInfo(typing.TypedDict):
    '''Grid tiling configuration of the world grid.'''
    gid: str
    tile_size_x: int
    tile_size_y: int
    tile_overlap_x: int
    tile_overlap_y: int
    tile_step_x: int
    tile_step_y: int

# ----- io_conventions
class _IOConventions(typing.TypedDict):
    '''Block IO format, key names, shapes, dtypes, and ignore index.'''
    block_format: str
    keys: _IOKeys
    shapes: _IOShapes
    dtypes: _IODtypes
    ignore_index: int

class _IOKeys(typing.TypedDict):
    '''NPZ key names for block serialization.'''
    image_key: str
    label_key: str
    valid_mask_key: str
    meta_key: str

class _IOShapes(typing.TypedDict):
    '''Logical ordering conventions for tensors.'''
    image_order: str  # e.g., 'C,H,W'
    label_order: str  # e.g., 'L,H,W'

class _IODtypes(typing.TypedDict):
    '''Dtypes for serialized arrays.'''
    image: str
    label: str
    valid_mask: str

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
    head_parent: dict[str, str | None]
    head_parent_cls: dict[str, int | None]

# ----- normalization
class _NormalizationInfo(typing.TypedDict):
    '''Normalization method and referenced stats files.'''
    method: str
    fit_stats_file: str
    test_stats_file: str # file can be non-existence

# ----- splits
class _SplitsInfo(typing.TypedDict):
    '''Train/val/test block references with recorded checksums.'''
    train_blocks: str
    val_blocks: str
    test_blocks: str # file can be non-existence

# ----- training_stats
class _TrainingStatsInfo(typing.TypedDict):
    '''Global and train-only class count references.'''
    class_counts_train: str
    class_counts_global: str

# ----- checksums
class _Checksums(typing.TypedDict):
    '''Artifacts checksums.'''
    train_blocks: str
    val_blocks: str
    test_blocks: str # empty string when test set is absent
    class_counts_train: str
    class_counts_global: str
    fit_stats_file: str
    test_stats_file: str # empty string when test set is absent
