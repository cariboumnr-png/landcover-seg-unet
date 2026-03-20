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
Utilities for stratified selection of validation/test blocks.

This module provides:
    - A validation-first (and optional test) selection that matches
      global class proportions using a simple greedy strategy,
    - Block-atomic splits to prevent spatial leakage across splits,
    - Deterministic ordering by block mass for stable, reusable splits,
    - Optional inverse-frequency weighting to emphasize rare classes,
    - Budgets defined by block count; set test_ratio=0.0 to skip test.

Typical workflow:
    1) Prepare per-block class counts with shape [n_blocks, n_classes],
    2) Call `stratified_splitter` to select validation (and test),
    3) Assign all remaining blocks to training,
    4) Apply any training-time oversampling or loss reweighting
       independently of the split.

The main entry point is `stratified_splitter`, which returns a structured
`_SplitIndices` containing sorted `train`, `val`, and `test` index lists.
'''

# standard imports
import dataclasses
import typing
import math
# third-party imports
import numpy
# local imports
import landseg.core.alias as alias

# ------------------------------private dataclass------------------------------
@dataclasses.dataclass
class _SplitIndices:
    '''Container for block indices from a split.'''
    train: typing.Iterable[int]
    val: typing.Iterable[int]
    test: typing.Iterable[int]

    def as_lists(self) -> typing.Self:
        '''In-place set each indices to a sorted list.'''

        self.train = sorted(list(self.train))
        self.val = sorted(list(self.val))
        self.test = sorted(list(self.test))
        return self

# -------------------------------Public Function-------------------------------
def stratified_splitter(
    counts: alias.Int64Array,
    val_ratio: float = 0.15,
    test_ratio: float = 0.0,
    weight_mode: typing.Literal['none', 'inverse'] = 'inverse',
) -> _SplitIndices:
    '''
    Stratified three-way splitter to generate train/val/test splits.

    Pick validation (and optionally test) blocks to match global class
    proportions, and the remaining blocks are assigned to training.

    Args:
        counts: Per-block integer class counts ([n_blocks, n_classes]).
        val_ratio: Target fraction of blocks in validation.
        test_ratio: Target fraction of blocks in test. Use 0.0 to skip.
        weight_mode: 'inverse' upweights rare classes via
            1 / max(gloabl_k, 1).

    Returns:
        Indices of each split in a container `_SplitIndices`.
    '''

    # input validations
    if counts.ndim != 2:
        raise ValueError('counts must be 2D [n_blocks, n_classes].')
    if counts.shape[0] == 0 or counts.shape[1] == 0:
        raise ValueError('counts must be non-empty.')
    if not numpy.issubdtype(counts.dtype, numpy.integer):
        raise ValueError('counts must be integer dtype.')
    if val_ratio < 0 or test_ratio < 0:
        raise ValueError('ratios must be non-negative.')
    if val_ratio + test_ratio >= 1.0:
        raise ValueError('val_ratio + test_ratio must be < 1.0.')

    # init a returning container
    idx = _SplitIndices(set(), set(), set())

    # first dim as the number of blocks
    n_blks = counts.shape[0]

    # get block budgets
    val_budget, test_budget = _block_budgets(n_blks, val_ratio, test_ratio)

    # get targets
    global_counts = counts.sum(axis=0).astype(numpy.float64)
    weights = _get_weights(global_counts, weight_mode)
    target_val = global_counts * val_ratio
    target_test = global_counts * test_ratio

    # deterministic order: larger total-count blocks first
    full = numpy.argsort(counts.sum(axis=1))[::-1].tolist() # reversed

    # validation selection
    idx.val = _pick_subset(counts, full, val_budget, target_val, weights)
    remain = [i for i in range(n_blks) if i not in idx.val] # remainder

    # test selection - returns empty set if budget is zero
    idx.test = _pick_subset(counts, remain, test_budget, target_test, weights)

    # rest goes to train
    idx.train = set(range(n_blks)) - (idx.val | idx.test)

    # return as lists
    return idx.as_lists()

# ---------- internal helpers ----------
def _block_budgets(
    n: int,
    val_ratio: float,
    test_ratio: float
) -> tuple[int, int]:
    '''Compute integer budgets for val and test by block count.'''

    val_budget = int(round(val_ratio * n))
    val_budget = max(0, min(val_budget, n))
    rem = n - val_budget
    test_budget = int(round(test_ratio * n))
    test_budget = max(0, min(test_budget, rem))
    return val_budget, test_budget

def _get_weights(
    global_counts: alias.Float64Array,
    mode: typing.Literal['none', 'inverse']
) -> alias.Float64Array:
    '''Build per-class weights for deviation scoring.'''

    safe = numpy.maximum(global_counts, 1.0)
    if mode == 'none':
        return numpy.ones_like(global_counts, dtype=numpy.float64)
    if mode == 'inverse':
        return 1.0 / safe
    raise ValueError("weight_mode must be 'none' or 'inverse'.")

def _pick_subset(
    counts: alias.Int64Array,
    candidates: list[int],
    budget: int,
    target: alias.Float64Array,
    weights: alias.Float64Array,
) -> set[int]:
    '''Greedy selection that minimizes weighted L1 dev. to target.'''

    # early exit
    if budget <= 0 or not candidates:
        return set()

    # init vars
    selected: list[int] = []
    current = numpy.zeros(counts.shape[1], dtype=numpy.float64)
    remaining = candidates[:]

    # selection process
    while len(selected) < budget and remaining:
        best_i = None
        best_score = math.inf
        for i in remaining:
            vec = counts[i].astype(numpy.float64)
            score = numpy.sum(weights * numpy.abs((current + vec) - target))
            if score < best_score:
                best_score = score
                best_i = i
        # exit if no more good blocks
        if best_i is None:
            break

        # add/remove best
        selected.append(best_i)
        remaining.remove(best_i)
        current += counts[best_i].astype(numpy.float64)

    return set(selected)
