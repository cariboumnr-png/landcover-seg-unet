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

'''Unit tests for the parallel execution framework (multip.py).'''

# local imports
import landseg.utils.multip as multip


def test_parallel_executor_threads():
    '''
    Given: A list of tasks.
    When: Running ParallelExecutor in multi-threading mode.
    Then: Return the expected mapped calculation results.
    '''
    executor = multip.ParallelExecutor(
        max_workers=2,
        use_threads=True,
        show_progress=False
    )
    jobs = [(_square, (i,), {}) for i in range(5)]
    results = executor.run(jobs)

    assert results == [0, 1, 4, 9, 16]


def test_parallel_executor_processes():
    '''
    Given: A list of tasks.
    When: Running ParallelExecutor in multi-processing mode.
    Then: Return the expected mapped calculation results.
    '''
    executor = multip.ParallelExecutor(
        max_workers=2,
        use_threads=False,
        show_progress=False
    )
    jobs = [(_square, (i,), {}) for i in range(5)]
    results = executor.run(jobs)

    assert results == [0, 1, 4, 9, 16]


def test_parallel_executor_captures_exceptions():
    '''
    Given: A list of tasks containing a failing function.
    When: Running ParallelExecutor.
    Then: Return the valid results and catch/serialize the failure
        traceback.
    '''
    executor = multip.ParallelExecutor(
        max_workers=2,
        use_threads=True,
        show_progress=False
    )
    jobs = [
        (_square, (2,), {}),
        (_failing_func, (3,), {}),
        (_square, (4,), {})
    ]
    results = executor.run(jobs)

    assert results[0] == 4
    assert isinstance(results[1], dict)
    assert 'error' in results[1]
    assert 'Failed with input 3' in results[1]['error']
    assert 'ValueError' in results[1]['traceback']
    assert results[2] == 16


def test_parallel_executor_with_progress(mocker):
    '''
    Given: A list of tasks.
    When: Running ParallelExecutor with show_progress enabled.
    Then: Render progress bar via tqdm during execution.
    '''
    mock_tqdm = mocker.patch(
        'tqdm.tqdm',
        side_effect=lambda it, **kwargs: list(it)
    )

    executor = multip.ParallelExecutor(
        max_workers=2,
        use_threads=True,
        show_progress=True,
        desc='Testing'
    )
    jobs = [(_square, (i,), {}) for i in range(3)]
    results = executor.run(jobs)

    assert results == [0, 1, 4]
    mock_tqdm.assert_called_once()


def _square(x):
    return x * x


def _failing_func(x):
    raise ValueError(f'Failed with input {x}')
