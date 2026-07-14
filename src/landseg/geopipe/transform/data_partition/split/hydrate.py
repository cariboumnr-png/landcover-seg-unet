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
Hydrating training blocks (greedy) to rebalance class counts.

The main routine incrementally accepts score-sorted blocks that improve
progress toward target class ratios using diminishing-returns reward, and
stops early when all targets are met or recent additions skew too heavily
toward non-target classes.

Includes helpers for priority computation, reward checking, and skew
detection.
'''

# standard imports
import dataclasses
import collections

# global hyperparameters
EPS = 1e-6          # safety
LOOKBACK = 20       # rolling window size for skew tracking

#
@dataclasses.dataclass(frozen=True)
class HydrationResults:
    '''Container for hydration results.'''
    hydrated_train_blocks: list[tuple[int, int]] # as coordinates
    hydrated_class_count: list[int]
    stop_message: str # for downstream logging


# -------------------------------Public Function-------------------------------
def hydrate_train_split(
    current_class_count: list[int],
    candidates: dict[tuple[int, int], list[int]],
    *,
    target_ratios: dict[int, float],
    max_skew_rate: float
) -> HydrationResults:
    '''
    Greedily select candidate blocks to move class counts toward targets.

    The function scans candidates in their incoming (score-sorted) order
    and accepts blocks that yield positive diminishing-returns reward:
    reward = sum_k priority_k * block_k / (current_k + eps)
    where priority_k is the normalized shortfall for class k.

    Early stop:
    * All targeted classes meet their target totals.
    * Recent additions skew toward non-targets beyond `skew_tol`.

    Module constants:
    * EPS: numerical stability in diminishing returns.
    * LOOKBACK: length of rolling window for skew detection.

    Args:
        current_class_count: Current per-class counts (length defines
            number of classes).
        candidates: Sequence of (name, per-class counts) aligned to
            class indices.
        target_ratios: Multipliers per class index. Classes with
            ratio > 1 are treated as targets to grow; classes with
            ratio ≤ 1 are treated as non-targets.
        max_skew_rate: Max recent non-target/target gain ratio before
            stopping. e.g., `max_skew_rate=5.0` means stop if non-targets
            grow 5x faster recently in the past `LOOKBACK` blocks.

    Returns:
        (selected_names, updated_counts, stop_reason).
    '''
    # sanity check
    assert all(len(c) == len(current_class_count) for c in candidates.values())

    # target total for each class is initial * ratio (default ratio = 1.0)
    targets = [
        current_class_count[i] * target_ratios.get(i, 1.0)
        for i in range(len(current_class_count))
    ]
    target_set = {i for i, r in target_ratios.items() if r > 1.0}

    # early exits
    if not target_set:
        return HydrationResults(
            hydrated_train_blocks=[],
            hydrated_class_count=[],
            stop_message='no hydration requested'
        )

    if all(current_class_count[i] + EPS >= targets[i] for i in target_set):
        return HydrationResults(
            hydrated_train_blocks=[],
            hydrated_class_count=[],
            stop_message='hydration targets already satisfied'
        )

    # selected blocks
    selected: list[tuple[int, int]] = []
    # rolling history of (target_gain, non_target_gain)
    recent = collections.deque[tuple[int, int]](maxlen=LOOKBACK)

    # init priorities
    priorities = _priorities(targets, current_class_count, EPS)
    # iterate in given (score-sorted) order
    msg = 'iterated all candidate blocks'
    for coords, blk_count in candidates.items():

        # compute reward
        if _no_reward(priorities, blk_count, current_class_count):
            continue

        # prospective skew check (without committing the block)
        tgt_gain = sum(blk_count[i] for i in target_set)
        non_gain = sum(blk_count) - tgt_gain
        stopped, msg = _skew_stop(tgt_gain, non_gain, recent, max_skew_rate)
        if stopped:
            break

        # accept block
        selected.append(coords)

        # updates and tracking
        current_class_count = [
            int(a + b) for a, b in zip(current_class_count, blk_count)
        ]
        recent.append((tgt_gain, non_gain))
        priorities = _priorities(targets, current_class_count, EPS)

        # stop if all targets are met
        if all(current_class_count[i] + EPS >= targets[i] for i in target_set):
            msg = 'stop searching due to [all targets met]'
            break

    return HydrationResults(
        hydrated_train_blocks=selected,
        hydrated_class_count=current_class_count,
        stop_message=msg
    )


def _priorities(
    target_ratios: list[float],
    current_counts: list[int],
    eps: float
) -> list[float]:
    '''Compute normalized shortfall priorities (range 0..1).'''

    p: list[float] = []
    number_class = len(target_ratios)
    for i in range(number_class):
        short = max(0.0, target_ratios[i] - current_counts[i])
        p.append(short / (target_ratios[i] + eps))
    return p


def _no_reward(priorities, blk_count, current_count) -> bool:
    '''Return True when the block yields no reward toward targets.'''

    reward = 0.0
    k = len(priorities)
    assert k == len(blk_count) == len(current_count) # sanity
    for i in range(k):
        # skip non-ops classes
        if priorities[i] <= 0.0 or blk_count[i] == 0:
            continue
        reward += priorities[i] * (blk_count[i] / (current_count[i] + EPS))
    # skip blocks that do not advance targets
    if reward <= 0.0:
        return True
    return False


def _skew_stop(
    target_gain: int,
    non_target_gain: int,
    recent: collections.deque[tuple[int, int]],
    skew_tol: float
) -> tuple[bool, str]:
    '''Decide if recent additions skew too far toward non-targets.'''

    # compute rolling totals as if we add this block
    roll_tgt = sum(t for t, _ in recent) + target_gain
    roll_non = sum(n for _, n in recent) + non_target_gain

    # if recent additions would skew back toward non-targets, stop search
    if roll_tgt == 0 and roll_non > 0:
        stop_reason = (
            'skew stop: recent additions add non-targets but yield no '
            'target-class gain'
        )
        return True, stop_reason
    ratio = (roll_non / roll_tgt) if roll_tgt > 0 else float('inf')
    if roll_tgt > 0 and ratio > skew_tol:
        stop_reason = (
            f'skew stop: non-target/target gain ratio {ratio:.2f} exceeds '
            f'tolerance {skew_tol:.2f}'
        )
        return True, stop_reason
    return False, ''
