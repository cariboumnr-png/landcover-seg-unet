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

'''Unit tests for stratified splitter logic (stratify.py).'''

# third-party imports
import numpy
import pytest
# local imports
import landseg.geopipe.transform.data_partition.split.stratify as stratify


# ----- `stratified_splitter` tests
def test_stratified_splitter_success():
    # base counts: 10 blocks, 3 classes
    # block coords mapping to class count lists
    base_counts = {
        (0, 0): [10, 0, 1],
        (0, 1): [5, 5, 0],
        (1, 0): [2, 10, 2],
        (1, 1): [0, 1, 15],
        (2, 0): [12, 1, 1],
        (2, 1): [8, 8, 0],
        (3, 0): [1, 15, 2],
        (3, 1): [0, 0, 20],
        (4, 0): [15, 2, 2],
        (4, 1): [10, 10, 0],
    }

    result = stratify.stratified_splitter(
        base_counts,
        val_ratio=0.2,
        test_ratio=0.1,
        weight_mode='inverse'
    )

    # 10 blocks total: val budget = 2 (round(0.2*10)), test budget = 1 (round(0.1*10))
    # train budget = 7
    assert len(result.train) == 7
    assert len(result.val) == 2
    assert len(result.test) == 1

    # verify no overlapping elements
    train_set = set(result.train)
    val_set = set(result.val)
    test_set = set(result.test)
    assert not (train_set & val_set)
    assert not (train_set & test_set)
    assert not (val_set & test_set)

    # verify totals sum up
    global_total = numpy.array(list(base_counts.values())).sum(axis=0)
    train_total = sum(numpy.array(base_counts[c]) for c in result.train)
    val_total = sum(numpy.array(base_counts[c]) for c in result.val)
    test_total = sum(numpy.array(base_counts[c]) for c in result.test)
    sum_totals = train_total + val_total + test_total
    assert numpy.array_equal(global_total, sum_totals)

    # verify dataclass statistics properties
    assert isinstance(result.class_counts, dict)
    assert isinstance(result.class_distributions, dict)
    assert result.class_counts['global'] == list(global_total)
    assert sum(result.class_distributions['global']) == pytest.approx(1.0)


def test_stratified_splitter_invalid_inputs():
    # counts must be non-empty
    with pytest.raises(ValueError, match='Counts must be non-empty'):
        stratify.stratified_splitter({}, val_ratio=0.1)

    # ratios sum must be < 1.0
    base_counts = {(0, 0): [1, 2]}
    with pytest.raises(ValueError, match='val \\+ test ratios must be < 1'):
        stratify.stratified_splitter(base_counts, val_ratio=0.5, test_ratio=0.5)

    # ratios must be >= 0
    with pytest.raises(ValueError, match='Ratios must be > 0'):
        stratify.stratified_splitter(base_counts, val_ratio=-0.1)


def test_stratified_splitter_reproducibility():
    base_counts = {
        (i, j): [i + 1, j + 2] for i in range(5) for j in range(5)
    }

    res1 = stratify.stratified_splitter(base_counts, val_ratio=0.2, test_ratio=0.2)
    res2 = stratify.stratified_splitter(base_counts, val_ratio=0.2, test_ratio=0.2)

    assert res1.train == res2.train
    assert res1.val == res2.val
    assert res1.test == res2.test
