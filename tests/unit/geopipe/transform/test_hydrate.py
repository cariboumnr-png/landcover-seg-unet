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

# pylint: disable=missing-function-docstring
# pylint: disable=protected-access

'''Unit tests for training split hydration logic (hydrate.py).'''

# third-party imports
import pytest
# local imports
import landseg.geopipe.transform.data_partition.split.hydrate as hydrate


# ----- `hydrate_train_split` tests
def test_hydrate_train_split_no_targets():
    # no class ratio > 1.0 (means no hydration is requested)
    current_counts = [100, 200]
    candidates = {
        (0, 0): [10, 20],
        (0, 1): [5, 30]
    }
    ratios = {0: 1.0, 1: 0.8}

    result = hydrate.hydrate_train_split(
        current_counts,
        candidates,
        target_ratios=ratios,
        max_skew_rate=5.0
    )

    assert len(result.hydrated_train_blocks) == 0
    assert len(result.hydrated_class_count) == 0
    assert result.info == 'no hydration requested'


def test_hydrate_train_split_targets_satisfied():
    # targeted class is 0 (current count is 0, so target count is 0 * 1.5 = 0, which is satisfied)
    current_counts = [0, 200]
    candidates = {
        (0, 0): [10, 20]
    }
    ratios = {0: 1.5}

    result = hydrate.hydrate_train_split(
        current_counts,
        candidates,
        target_ratios=ratios,
        max_skew_rate=5.0
    )

    assert len(result.hydrated_train_blocks) == 0
    assert len(result.hydrated_class_count) == 0
    assert result.info == 'hydration targets already satisfied'


def test_hydrate_train_split_greedy_selection():
    # target class 0 to grow by 1.5x (from 100 to 150)
    current_counts = [100, 200]
    candidates = {
        (0, 0): [30, 10], # advances target 0 by 30
        (0, 1): [0, 50],  # does not advance target 0
        (1, 0): [20, 5],  # advances target 0 by 20
    }
    ratios = {0: 1.5}

    result = hydrate.hydrate_train_split(
        current_counts,
        candidates,
        target_ratios=ratios,
        max_skew_rate=5.0
    )

    # should select block (0,0) and block (1,0), skip (0,1)
    assert result.hydrated_train_blocks == [(0, 0), (1, 0)]
    # final counts should be [100+30+20, 200+10+5] = [150, 215]
    assert result.hydrated_class_count == [150, 215]
    assert 'stop searching due to [all targets met]' in result.info


def test_hydrate_train_split_skew_stop():
    # target class 0 to grow from 10 to 50
    current_counts = [10, 10]
    candidates = {
        (0, 0): [1, 20], # advances target 0 by 1, but adds 20 of class 1 (skew ratio = 20)
    }
    ratios = {0: 5.0}

    result = hydrate.hydrate_train_split(
        current_counts,
        candidates,
        target_ratios=ratios,
        max_skew_rate=5.0 # skew tolerance is 5.0
    )

    # should stop because skew ratio (20/1 = 20) exceeds tolerance (5.0)
    assert len(result.hydrated_train_blocks) == 0
    assert 'skew stop' in result.info
