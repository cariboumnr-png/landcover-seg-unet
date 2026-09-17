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
Utilities for stratified selection of validation and test blocks.

Selects blocks to match global class proportions using greedy deviation
scoring, preserving block atomicity to prevent spatial leakage.

Public APIs:
    - SplitsResult: container for split coordinates and class stats.
    - stratified_splitter: stratified splitter for train/val/test.
'''

# standard imports
import dataclasses
import typing
# third-party imports
import numpy
# local imports
import landseg.geopipe.alias as geo_alias


# ----- public dataclasses
@dataclasses.dataclass(frozen=True)
class SplitsResult:
    '''Container for split coordinates and derived class statistics.'''
    train: geo_alias.CoordsList
    val: geo_alias.CoordsList
    test: geo_alias.CoordsList
    global_class_count: tuple[int, ...]
    train_class_count: tuple[int, ...]
    val_class_count: tuple[int, ...]
    test_class_count: tuple[int, ...]

    @property
    def class_counts(self) -> geo_alias.ClassCounts:
        '''Return per-split class counts.'''
        return {
            'global': list(self.global_class_count),
            'train': list(self.train_class_count),
            'val': list(self.val_class_count),
            'test': list(self.test_class_count),
        }

    @property
    def class_distributions(self) -> dict[str, list[float]]:
        '''Return per-split normalized class distributions.'''
        def _safe_distribution(cc: typing.Sequence[int]) -> list[float]:
            total = sum(cc)
            if total <= 0:
                return [0.0 for _ in cc]
            return [float(count / total) for count in cc]

        return {
            split_name: _safe_distribution(class_count)
            for split_name, class_count in self.class_counts.items()
        }


# ----- public functions
def stratified_splitter(
    base_counts: geo_alias.CoordClassCounts,
    *,
    val_ratio: float = 0.15,
    test_ratio: float = 0.0,
    weight_mode: typing.Literal['none', 'inverse'] = 'inverse',
) -> SplitsResult:
    '''
    Stratified three-way splitter to generate train/val/test splits.

    Picks validation and optionally test blocks to match global class
    proportions. Remaining blocks are assigned to training.

    Args:
        base_counts:
            per-block integer class counts indexed by coordinates.
        val_ratio:
            target fraction of blocks in validation.
        test_ratio:
            target fraction of blocks in test (use 0.0 to skip).
        weight_mode:
            weighting strategy for deviation scoring ('none' or
            'inverse').

    Returns:
        SplitsResult:
            split coordinates, class counts, and distributions.
    '''
    # base class counts dict to array
    counts: geo_alias.Int64Array = numpy.array(
        list(base_counts.values()), dtype=numpy.int64
    )
    coords: geo_alias.CoordsList = list(base_counts.keys())
    global_counts: geo_alias.Int64Array = counts.sum(axis=0).astype(numpy.int64)
    n_blocks = counts.shape[0]

    _validate_inputs(counts, val_ratio, test_ratio)

    val_budget, test_budget = _get_budgets(n_blocks, val_ratio, test_ratio)

    weights: geo_alias.Float64Array = _get_weights(global_counts, weight_mode)

    # deterministic order: larger total-count blocks first.
    all_sorted = numpy.argsort(counts.sum(axis=1))[::-1].tolist()
    val_idx = _pick_subset(
        counts=counts,
        candidates=all_sorted,
        budget=val_budget,
        target=(global_counts * val_ratio).astype(numpy.float64),
        weights=weights,
    )

    test_idx = _pick_subset(
        counts=counts,
        candidates=[i for i in range(n_blocks) if i not in val_idx],
        budget=test_budget,
        target=(global_counts * test_ratio).astype(numpy.float64),
        weights=weights,
    )

    train_idx = set(range(n_blocks)) - val_idx - test_idx

    # sanity checks on leakage before return
    if train_idx & val_idx or train_idx & test_idx or val_idx & test_idx:
        raise ValueError('Data leaked between split partitions!')

    return SplitsResult(
        train=[coords[i] for i in sorted(train_idx)],
        val=[coords[i] for i in sorted(val_idx)],
        test=[coords[i] for i in sorted(test_idx)],
        global_class_count=_sum_class_counts(counts, range(n_blocks)),
        train_class_count=_sum_class_counts(counts, train_idx),
        val_class_count=_sum_class_counts(counts, val_idx),
        test_class_count=_sum_class_counts(counts, test_idx),
    )


# ----- private helpers
def _validate_inputs(
    counts: geo_alias.IntArray,
    val_ratio: float,
    test_ratio: float,
) -> None:
    '''Validate splitter inputs.'''
    if counts.shape[0] == 0 or counts.shape[1] == 0:
        raise ValueError('Counts must be non-empty.')

    if counts.ndim != 2:
        raise ValueError(
            f'Counts must be 2D [n_blocks, n_classes], got: {counts.ndim}'
        )

    if not numpy.issubdtype(counts.dtype, numpy.integer):
        raise ValueError(
            f'Counts must be of integer dtype, got: {counts.dtype}'
        )

    if val_ratio < 0 or test_ratio < 0:
        raise ValueError(
            f'Ratios must be > 0, got val: {val_ratio}, test: {test_ratio}'
        )

    if val_ratio + test_ratio >= 1.0:
        raise ValueError(
            f'val + test ratios must be < 1, got: {val_ratio + test_ratio}'
        )


def _get_budgets(
    n: int,
    val_ratio: float,
    test_ratio: float,
) -> tuple[int, int]:
    '''Compute integer budgets for validation and test splits.'''
    val_budget = int(round(val_ratio * n))
    val_budget = max(0, min(val_budget, n))

    remaining = n - val_budget

    test_budget = int(round(test_ratio * n))
    test_budget = max(0, min(test_budget, remaining))

    return val_budget, test_budget


def _get_weights(
    global_counts: geo_alias.Int64Array,
    mode: typing.Literal['none', 'inverse'],
) -> geo_alias.Float64Array:
    '''Build per-class weights for deviation scoring.'''
    safe_counts = numpy.maximum(global_counts.astype(numpy.float64), 1.0)

    if mode == 'none':
        return numpy.ones_like(global_counts, dtype=numpy.float64)

    if mode == 'inverse':
        return 1.0 / safe_counts

    raise ValueError('weight_mode must be "none" or "inverse".')


def _pick_subset(
    counts: geo_alias.Int64Array,
    candidates: list[int],
    budget: int,
    target: geo_alias.Float64Array,
    weights: geo_alias.Float64Array,
) -> set[int]:
    '''Greedily select blocks that minimize weighted L1 to target.'''
    if budget <= 0 or not candidates:
        return set()

    selected: list[int] = []
    current: geo_alias.Float64Array = numpy.zeros(
        counts.shape[1], dtype=numpy.float64
    )
    remaining = candidates[:]

    while len(selected) < budget and remaining:
        best_i = None
        best_score = float('inf')

        for i in remaining:
            vec = counts[i].astype(numpy.float64)
            score = numpy.sum(weights * numpy.abs((current + vec) - target))

            if score < best_score:
                best_score = score
                best_i = i

        if best_i is None:
            break

        selected.append(best_i)
        remaining.remove(best_i)
        current += counts[best_i].astype(numpy.float64)

    return set(selected)


def _sum_class_counts(
    counts: geo_alias.Int64Array,
    indices: typing.Iterable[int],
) -> tuple[int, ...]:
    '''Sum class counts for selected block indices.'''
    idx = list(indices)
    if not idx:
        return tuple(0 for _ in range(counts.shape[1]))
    return tuple(int(x) for x in numpy.sum(counts[idx], axis=0))
