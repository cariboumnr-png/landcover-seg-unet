# =========================================================================== #
#            Copyright © His Majesty the King in right of Ontario,            #
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
Dataset-level preparation schema definitions.

Defines structured TypedDict schemas for dataset-wide preparation
metadata, capturing splits, checksums, label counts, and image stats.

Public APIs:
    - `ImageBandStats`: Online band running statistics.
    - `PreparedSchema`: Dataset-wide preparation schema.
    - `TargetHeadsSchema`: Target heads hierarchy and reclassification.
'''

# standard imports
from __future__ import annotations
import typing


SCHEMA_ID = 'prepared_schema/v0.1.0'


# ----- public types
class PreparedSchema(typing.TypedDict):
    '''
    Dataset-wide preparation schema.

    Captures metadata and statistics generated during preparation,
    including splits, checksums, label counts, and image stats.

    Schema: SCHEMA_ID = 'prepared_schema/v0.1.0'
    '''
    schema_id: str
    creation_time: str
    artifacts: dict[str, str]
    checksums: dict[str, str]
    train_blocks: dict[str, str]
    val_blocks: dict[str, str]
    test_blocks: dict[str, str]
    label_stats: dict[str, list[int]]
    image_stats: dict[str, ImageBandStats]
    image_array_key: str
    label_array_key: str
    heads: typing.NotRequired[TargetHeadsSchema]


class ImageBandStats(typing.TypedDict):
    '''
    Statistical summary for a single image band.

    Fields:
        total_count: Total number of valid pixels observed.
        current_mean: Running mean of pixel values.
        accum_m2: Accumulated sum of squared differences from mean.
        std: Standard deviation of pixel values.
    '''
    total_count: int
    current_mean: float
    accum_m2: float
    std: float


class TargetHeadsSchema(typing.TypedDict):
    '''
    Resolved multi-head target hierarchy and reclassification specs.

    Fields:
        head_names: Ordered list of target head names.
        head_parent: Mapping of child head to parent group head name.
        head_parent_cls: Mapping of child head to parent class index.
        num_classes: Mapping of head names to class counts.
        class_names: Mapping of head names to class label names.
        ignore_classes: Mapping of head to ignored class indices.
    '''
    head_names: list[str]
    head_parent: dict[str, str | None]
    head_parent_cls: dict[str, int | None]
    num_classes: dict[str, int]
    class_names: dict[str, list[str]]
    ignore_classes: dict[str, list[int]]
