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
Greedy training block hydration for class rebalancing.

Incrementally accepts score-sorted candidate blocks that improve
progress toward target class ratios using diminishing-returns reward,
and stops early when all targets are met or additions skew toward
non-target classes.

Public APIs:
    - HydrationResults: container for hydrated blocks and counts.
    - hydrate_train_split: hydrate training split toward targets.
'''

# standard imports
import dataclasses
import collections

# aliases
field = dataclasses.field

# global hyperparameters
EPS = 1e-6          # safety
LOOKBACK = 20       # rolling window size for skew tracking


# ----- public dataclasses
@dataclasses.dataclass(frozen=True)
class HydrationResults:
    '''Container for hydrated block coordinates and class counts.'''
    hydrated_train_blocks: list[tuple[int, int]] = field(default_factory=list)
    hydrated_class_count: list[int] = field(default_factory=list)
    info: str = 'no hydration requested' # for downstream logging


# ----- public functions
def hydrate_train_split(
    current_class_count: list[int],
    candidates: dict[tuple[int, int], list[int]],
    *,
    target_ratios: dict[int, float],
    max_skew_rate: float,
) -> HydrationResults:
    '''
    Select candidate blocks to move class counts toward targets.

    Scans candidates in score-sorted order and accepts blocks that yield
    positive diminishing-returns reward, stopping when targets are met
    or recent additions skew excessively toward non-target classes.

    Args:
        current_class_count:
            current per-class pixel counts across the training split.
        candidates:
            mapping of candidate block coordinates to class counts.
        target_ratios:
            target multipliers per class index (ratio > 1.0 to grow).
        max_skew_rate:
            maximum allowed ratio of non-target to target gain.

    Returns:
        HydrationResults:
            selected blocks, updated class counts, and termination info.
    '''
    # sanity check
    assert all(len(c) == len(current_class_count) for c in candidates.values())

    # target total for each class is initial * ratio (default = 1.0)
    targets = [
        current_class_count[i] * target_ratios.get(i, 1.0)
        for i in range(len(current_class_count))
    ]
    target_set = {i for i, r in target_ratios.items() if r > 1.0}

    # early exits
    if not target_set:
        return HydrationResults()

    if all(current_class_count[i] + EPS >= targets[i] for i in target_set):
        return HydrationResults(info='hydration targets already satisfied')

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
        info=msg
    )


# ----- private helpers
def _priorities(
    target_ratios: list[float],
    current_counts: list[int],
    eps: float,
) -> list[float]:
    '''Compute normalized shortfall priorities in range [0, 1].'''
    p: list[float] = []
    number_class = len(target_ratios)
    for i in range(number_class):
        short = max(0.0, target_ratios[i] - current_counts[i])
        p.append(short / (target_ratios[i] + eps))
    return p


def _no_reward(
    priorities: list[float],
    blk_count: list[int],
    current_count: list[int],
) -> bool:
    '''Return True when block yields no reward toward targets.'''
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

    # if additions would skew back toward non-targets, stop search
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
