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
TypedDict-based configuration schemas for dataprep, block building,
scoring, and process orchestration. Defines structured config interfaces
used throughout the dataprep pipeline.

Public APIs:
    - InputConfig: Paths for bimodal input rasters and config files.
    - OutputConfig: Paths to dataprep-generated artifacts.
    - PixelThresholdConfig: Thresholds for validating block pixel ratios.
    - ScoringConfig: Parameters for block scoring onlabel distribution.
    - IOConfig: Combined input/output configuration for block building.
    - BlockBuildingConfig: Full block-building configuration with pixel
      thresholds included.
    - ProcessConfig: Post-block-building configuration used for splitting,
      scoring, and aggregation.
    - DataprepConfigs: Composite configuration covering the entire
      dataprep workflow.
'''

# standard imports
import typing

class InputConfig(typing.TypedDict):
    '''Bimodal input rasters and config file paths.'''
    fit_input_img: str
    fit_input_lbl: str
    test_input_img: str | None
    test_input_lbl: str | None
    input_config: str

class OutputConfig(typing.TypedDict):
    '''Paths to artifacts generated during data preparation.'''
    fit_windows: str
    fit_blks_dir: str
    fit_all_blks: str
    fit_valid_blks: str
    fit_img_stats: str
    lbl_count_global: str
    blk_scores: str
    train_blks: str
    val_blks: str
    lbl_count_train: str
    test_windows: str
    test_blks_dir: str
    test_all_blks: str
    test_valid_blks: str
    test_img_stats: str

class PixelThresholdConfig(typing.TypedDict):
    '''Block validation pixel ratio thresholds.'''
    blk_thres_fit: float
    blk_thres_test: float

class ScoringConfig(typing.TypedDict):
    '''Block label representation scoring config.'''
    score_head: str
    score_alpha: float
    score_beta: float
    score_epsilon: float
    score_reward: tuple[int, ...]

class IOConfig(
    InputConfig,
    OutputConfig,
    typing.TypedDict
):
    '''Combined input/output paths for configuring a block builder.'''

class BlockBuildingConfig(
    InputConfig,
    OutputConfig,
    PixelThresholdConfig,
    typing.TypedDict
):
    '''Combined configuration for block building.'''

class ProcessConfig(
    OutputConfig,
    ScoringConfig,
    typing.TypedDict
):
    '''Post-block-building configuration for scoring and splitting.'''

class DataprepConfigs(
    InputConfig,
    OutputConfig,
    PixelThresholdConfig,
    ScoringConfig,
    typing.TypedDict,
):
    '''Composite configuration for the full dataprep workflow.'''
