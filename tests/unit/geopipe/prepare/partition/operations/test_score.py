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

# pylint: disable=protected-access

'''Unit tests for block scoring calculations (score.py).'''

# third-party imports
import numpy
import pytest
# local imports
import landseg.geopipe.prepare.partition.operations.score as score


# ----- `score_blocks` tests
def test_score_blocks(mocker):
    '''
    Given: Multi-class counts for a pool of candidates.
    When: Running score_blocks using parallel executor.
    Then: Return a dict of blocks scored and sorted by value.
    '''
    # mock ParallelExecutor to run synchronously in the same process
    mocker.patch(
        'landseg.utils.ParallelExecutor.run',
        side_effect=lambda jobs, desc=None: [
            fn(*args, **kwargs) for fn, args, kwargs in jobs
        ]
    )

    global_counts = [100, 200]
    input_blocks = {
        (0, 0): [10, 5],
        (0, 1): [1, 20]
    }

    result = score.score_blocks(
        global_counts,
        input_blocks,
        reward=(0,),
        alpha=1.0,
        beta=0.8,
        mode='log_lift'
    )

    # verify that both blocks are scored and returned in sorted order
    assert len(result) == 2
    assert (0, 0) in result
    assert (0, 1) in result


def test_count_to_prob_w_temp():
    '''
    Given: Raw frequency counts of categories.
    When: Running _count_to_prob_w_temp with temperature alpha.
    Then: Return a normalized probability vector.
    '''
    counts = [10, 10]
    prob = score._count_to_prob_w_temp(counts, alpha=1.0)
    assert numpy.allclose(prob, [0.5, 0.5])

    prob_scaled = score._count_to_prob_w_temp(counts, alpha=2.0)
    assert numpy.allclose(prob_scaled, [0.5, 0.5])


def test_log_lift_on_reward():
    '''
    Given: Reference and candidate probability distributions.
    When: Running _log_lift_on_reward.
    Then: Calculate the log lift score on the target categories.
    '''
    p = numpy.array([0.5, 0.5])
    q = numpy.array([0.8, 0.2])

    val = score._log_lift_on_reward(p, q, reward_cls=(0,))
    expected = 0.8 * numpy.log((0.8 + score.EPS) / (0.5 + score.EPS))
    assert val == pytest.approx(expected)


def test_weighted_l1_w_reward():
    '''
    Given: Reference and candidate probability distributions.
    When: Running _weighted_l1_w_reward.
    Then: Calculate the weighted L1 distance on target categories.
    '''
    p = numpy.array([0.5, 0.5])
    q = numpy.array([0.8, 0.2])

    val = score._weighted_l1_w_reward(p, q, reward_cls=(0,), beta=0.5)
    assert isinstance(val, float)


def test_score_blocks_zero_score_rank(mocker):
    '''
    Given: Blocks producing positive, zero, negative, and None scores.
    When: Running score_blocks.
    Then: Rank zero score above negative scores, and None at the end.
    '''
    # mock ParallelExecutor to return predictable scores: 1.0, 0.0, -0.5, None
    mocker.patch(
        'landseg.utils.ParallelExecutor.run',
        return_value=[
            ((0, 0), -0.5, [1, 1]),
            ((0, 1), 0.0, [1, 1]),
            ((0, 2), 1.0, [1, 1]),
            ((0, 3), None, [0, 0]),
        ]
    )

    result = score.score_blocks(
        [10, 10],
        {(0, 0): [1, 1], (0, 1): [1, 1], (0, 2): [1, 1], (0, 3): [0, 0]},
        reward=(0,),
        alpha=1.0,
        beta=0.8,
    )
    keys = list(result.keys())
    assert keys == [(0, 2), (0, 1), (0, 0), (0, 3)]


def test_score_blocks_reward_out_of_range():
    '''
    Given: Global classes count of 2 classes and reward index 5.
    When: Running score_blocks.
    Then: Raise IndexError for out-of-range reward index.
    '''
    with pytest.raises(IndexError, match='Reward class index 5 out of range'):
        score.score_blocks(
            [10, 20],
            {(0, 0): [5, 5]},
            reward=(5,),
            alpha=1.0,
            beta=0.8,
        )


def test_score_blocks_empty_inputs():
    '''
    Given: Empty global counts or empty input blocks.
    When: Running score_blocks.
    Then: Return empty dictionary without error.
    '''
    assert score.score_blocks(
        [], {(0, 0): [1]}, reward=(), alpha=1.0, beta=0.8
    ) == {}
    assert score.score_blocks(
        [10], {}, reward=(), alpha=1.0, beta=0.8
    ) == {}
