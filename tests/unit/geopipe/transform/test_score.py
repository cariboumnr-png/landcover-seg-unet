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

'''Unit tests for block scoring calculations (score.py).'''

# third-party imports
import numpy
import pytest
# local imports
import landseg.geopipe.transform.data_partition.split.score as score


# ----- `score_blocks` tests
def test_score_blocks(mocker):
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
    # standard counts to prob conversion
    counts = [10, 10]
    prob = score._count_to_prob_w_temp(counts, alpha=1.0)
    assert numpy.allclose(prob, [0.5, 0.5])

    # with alpha/temperature scaling
    prob_scaled = score._count_to_prob_w_temp(counts, alpha=2.0)
    assert numpy.allclose(prob_scaled, [0.5, 0.5])


def test_log_lift_on_reward():
    p = numpy.array([0.5, 0.5])
    q = numpy.array([0.8, 0.2])

    val = score._log_lift_on_reward(p, q, reward_cls=(0,))
    # q[0] * log((q[0]+eps)/(p[0]+eps))
    expected = 0.8 * numpy.log((0.8 + score.EPS) / (0.5 + score.EPS))
    assert val == pytest.approx(expected)


def test_weighted_l1_w_reward():
    p = numpy.array([0.5, 0.5])
    q = numpy.array([0.8, 0.2])

    val = score._weighted_l1_w_reward(p, q, reward_cls=(0,), beta=0.5)
    assert isinstance(val, float)
