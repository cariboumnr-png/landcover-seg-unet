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

'''Unit tests for training split hydration logic (hydrate.py).'''

# local imports
import landseg.geopipe.transform.data_partition.split.hydrate as hydrate


# ----- `hydrate_train_split` tests
def test_hydrate_train_split_no_targets():
    '''
    Given: Current class counts and candidate pools.
    When: Running hydrate_train_split with all target ratios <= 1.0.
    Then: Return early without adding any blocks.
    '''
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
    '''
    Given: Target class ratios that are already satisfied by counts.
    When: Running hydrate_train_split.
    Then: Return early with zero added blocks.
    '''
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
    '''
    Given: Candidates with varying target category densities.
    When: Running hydrate_train_split with a high variance target.
    Then: Select candidates greedily to satisfy target requirements.
    '''
    current_counts = [100, 200]
    candidates = {
        (0, 0): [30, 10],
        (0, 1): [0, 50],
        (1, 0): [20, 5],
    }
    ratios = {0: 1.5}

    result = hydrate.hydrate_train_split(
        current_counts,
        candidates,
        target_ratios=ratios,
        max_skew_rate=5.0
    )

    assert result.hydrated_train_blocks == [(0, 0), (1, 0)]
    assert result.hydrated_class_count == [150, 215]
    assert 'stop searching due to [all targets met]' in result.info


def test_hydrate_train_split_skew_stop():
    '''
    Given: Candidates containing high ratios of non-target classes.
    When: Running hydrate_train_split.
    Then: Terminate early if candidate skew exceeds limits.
    '''
    current_counts = [10, 10]
    candidates = {
        (0, 0): [1, 20],
    }
    ratios = {0: 5.0}

    result = hydrate.hydrate_train_split(
        current_counts,
        candidates,
        target_ratios=ratios,
        max_skew_rate=5.0
    )

    assert len(result.hydrated_train_blocks) == 0
    assert 'skew stop' in result.info
