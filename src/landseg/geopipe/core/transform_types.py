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
Typed dictionaries for JSON-based artifacts in `geopipe.transform`.

This module defines structured schemas used to represent intermediate
and final artifacts produced during dataset transformation. These
schemas standardize how dataset partitions, statistical summaries, and
global transformation metadata are serialized to JSON.

They ensure consistency, traceability, and validation of outputs across
the transformation pipeline.
'''

# standard imports
from __future__ import annotations
import typing

TRANSFORM_SCHEMA_ID = 'transform_schema/v1'

# ---------------------------------Public Type---------------------------------
class BlocksPartition(typing.TypedDict):
    '''
    Dataset partition mapping for block files.

    This structure defines how block artifacts are split across
    training, validation, and test sets.

    Fields:

    - **train**:
        Mapping of block identifiers to file paths for training data.
    - **val**:
        Mapping of block identifiers to file paths for validation data.
    - **test**:
        Mapping of block identifiers to file paths for test data.
    '''
    train: dict[str, str]
    val: dict[str, str]
    test: dict[str, str]

class ImageBandStats(typing.TypedDict):
    '''
    Statistical summary for a single image band.

    These statistics are accumulated across the dataset and can be used
    for normalization or analysis.

    Fields:

    - **total_count**:
        Total number of valid pixels observed.
    - **current_mean**:
        Running mean of pixel values.
    - **accum_m2**:
        Accumulated sum of squared differences from the mean (used for
        variance computation).
    - **std**:
        Standard deviation of pixel values.
    '''
    total_count: int
    current_mean: float
    accum_m2: float
    std: float

class TransformSchema(typing.TypedDict):
    '''
    Dataset-wide transformation schema.

    This structure captures metadata and statistics generated during the
    transformation stage, including dataset splits, checksums, and
    aggregated statistics for both labels and image bands.

    Fields:

    - **schema_version**:
        Version identifier for the transformation schema.
    - **creation_time**:
        Timestamp indicating when the schema was generated.
    - **artifacts**:
        Mapping of artifact names to file paths.
    - **checksums**:
        Mapping of artifact names to checksum values.
    - **train_blocks**:
        Mapping of training block identifiers to file paths.
    - **val_blocks**:
        Mapping of validation block identifiers to file paths.
    - **test_blocks**:
        Mapping of test block identifiers to file paths.
    - **label_stats**:
        Aggregated label class counts across the dataset.
    - **image_stats**:
        Aggregated per-band image statistics.
    - **image_array_key**:
        Key used to access image arrays in stored artifacts.
    - **label_array_key**:
        Key used to access label arrays in stored artifacts.
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
