# =========================================================================== #
#           Copyright © His Majesty the King in right of Ontario,           #
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
TypedDict definitions for dataset preparation reports and artifacts.

Defines schemas for data partitioning, block normalization, and schema
generation execution summaries and persisted preparation artifacts.

Public APIs:
    - `BlocksPartition`: TypedDict mapping block IDs across splits.
    - `DataPartitionReport`: Report for dataset splitting.
    - `ImageBandStats`: TypedDict for image band statistics.
    - `NormalizationReport`: Report for block materialization.
    - `PartitionSummary`: Summary of raw splits and hydration.
    - `PREPARED_SCHEMA_ID`: Constant string for prepared schema ID.
    - `PreparationReportSchema`: Root summary schema for prepare runs.
    - `PreparedSchema`: TypedDict for dataset preparation schema.
    - `SchemaReport`: Report for dataset schema generation.
    - `TargetHeadsSchema`: TypedDict for target heads hierarchy.
'''

# standard imports
from __future__ import annotations
import typing

PREPARED_SCHEMA_ID = 'prepared_schema/v1'


# ----- public types
class DataPartitionReport(typing.TypedDict):
    '''Execution report for dataset splitting and hydration.'''
    status: typing.Literal['loaded', 'created']
    duration_sec: float


class NormalizationReport(typing.TypedDict):
    '''Execution report for block normalization and materialization.'''
    status: typing.Literal['loaded', 'created']
    duration_sec: float
    unwanted_blocks_removed: int
    rebuild: bool
    stats_filepath: str


class SchemaReport(typing.TypedDict):
    '''Execution report for dataset schema generation.'''
    status: typing.Literal['loaded', 'created']
    duration_sec: float
    schema_filepath: str
    classes_mapped: list[str]


class PreparationReportSchema(typing.TypedDict):
    '''Root report mapping the entire data-prepare pipeline run.'''
    run_id: str
    timestamp: str
    status: typing.Literal['SUCCESS', 'FAILED']
    data_partition: DataPartitionReport | None
    normalization: NormalizationReport | None
    schema: SchemaReport | None


class BlocksPartition(typing.TypedDict):
    '''
    Dataset partition mapping for block files.

    This structure defines how block artifacts are split across
    training, validation, and test sets.
    '''
    train: dict[str, str]
    val: dict[str, str]
    test: dict[str, str]


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


class PreparedSchema(typing.TypedDict):
    '''
    Dataset-wide preparation schema.

    Captures metadata and statistics generated during preparation,
    including splits, checksums, label counts, and image stats.
    '''
    schema_version: str
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


class PartitionSummary(typing.TypedDict):
    '''
    Summary of raw splits and hydration statistics.

    Captures details of the initial block splitting and hydration.
    '''
    original_splits: _SplitsStats
    hydration: _HydrationStats


# ----- private types
class _SplitsStats(typing.TypedDict):
    '''Splits section.'''
    training: _PartitionStats
    validation: _PartitionStats
    testing: _PartitionStats


class _PartitionStats(typing.TypedDict):
    '''Details for each split.'''
    num_of_blocks: int
    class_count: list[int]
    class_distribution: list[float]


class _HydrationStats(typing.TypedDict):
    '''Hydration details.'''
    performed: bool
    focal_head: str
    stop_reason: str
    n_training_blocks: int
    n_training_blocks_change: int
    hydrated_class_count: list[int]
    class_count_change: list[int]
    hydrated_class_distribution: list[float]
    class_distribution_change: list[float]
