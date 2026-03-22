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
Hydrate training blocks for class rebalancing.
'''

# standard imports
import collections

def hydrate_train_split(
    current_class_count: list[int],
    candidates: list[tuple[str, list[int]]],
    target_ratios: dict[int, float],
    *,
    eps: float = 1e-6
):

    '''Greedy hydration of train blocks to approach target class ratios.

    The algorithm scans candidates in their given (score-sorted) order and
    greedily accepts blocks that improve progress toward the target class
    totals while applying diminishing returns:
        reward = sum_k(
            priority_k * block_count_k / (current_total_k + eps)
        )
    where priority_k is the normalized remaining shortfall for class k.

    Early stop conditions:
      * All targeted classes reach their target totals.
      * Recent additions start to skew toward non-target classes, i.e.,
        non-target gains outpace target gains beyond a tolerance.

    Assumptions:
      * `candidates` preserves insertion order (already sorted by score).
      * Each candidate has a 'count' list aligned to class indices.

    Returns:
      A tuple (selected_names, updated_counts).

    Notes:
      Constants can be tuned inside the function:
        - eps: numerical stability in diminishing returns.
        - skew_tol: how much faster non-targets may grow before stopping.
        - lookback: rolling window length for skew detection.
    '''


    # ---------- hyperparameters ----------
    skew_tol = 1.5     # stop if non-targets grow >10% faster recently
    lookback = 20       # rolling window size for skew tracking
    # -------------------------------------

    # Freeze initial counts to define fixed targets
    init_counts = list(current_class_count)
    k = len(init_counts) # number of classes
    # sanity check
    assert all(len(c) == k for _, c in candidates)
    current = _pad_counts(init_counts, k)

    # target total for each class is initial * ratio (default ratio = 1.0)
    targets = [current[i] * target_ratios.get(i, 1.0) for i in range(k)]
    target_set = {i for i, r in target_ratios.items() if r > 1.0} # only boost

    selected: list[str] = []
    # rolling history of (target_gain, non_target_gain)
    recent = collections.deque[tuple[int, int]](maxlen=lookback)

    # early exits
    # if no targets requested
    if not target_set:
        return selected, current
    # if already meeting all targets
    if all(current[i] + eps >= targets[i] for i in target_set):
        return selected, current

    # init priorities
    pri = _priorities(k, targets, current, eps)

    # iterate in given (score-sorted) order
    for blk_name, blk_counts in candidates:

        # compute diminishing-return reward toward targets
        reward = 0.0
        for i in range(k):
            if pri[i] <= 0.0 or blk_counts[i] == 0:
                continue
            reward += pri[i] * (blk_counts[i] / (current[i] + eps))

        # skip blocks that do not advance targets
        if reward <= 0.0:
            continue

        # prospective skew check (without committing the block)
        tgt_gain = sum(blk_counts[i] for i in range(k) if i in target_set)
        non_gain = sum(blk_counts[i] for i in range(k) if i not in target_set)

        # compute rolling totals as if we add this block
        roll_tgt = sum(t for t, _ in recent) + tgt_gain
        roll_non = sum(n for _, n in recent) + non_gain

        # if recent additions would skew back toward non-targets, stop search
        if roll_tgt == 0 and roll_non > 0:
            stop_reason = (
                'skew stop: recent additions add non-targets but yield no '
                'target-class gain'
            )
            print(stop_reason)
            break
        ratio = (roll_non / roll_tgt) if roll_tgt > 0 else float('inf')
        if roll_tgt > 0 and ratio > skew_tol:
            stop_reason = (
                f'skew stop: non-target/target gain ratio {ratio:.2f} exceeds '
                f'tolerance {skew_tol:.2f}'
            )
            print(stop_reason)
            break

        # accept block
        selected.append(blk_name)
        for i in range(k):
            current[i] += blk_counts[i]
        recent.append((tgt_gain, non_gain))

        # update priorities after state change
        pri = _priorities(k, targets, current, eps)

        # stop if all targets are met
        if all(current[i] + eps >= targets[i] for i in target_set):
            print('stop searching due to [all targets met]')
            break

    return selected, current

def _pad_counts(x: list[int], k: int) -> list[int]:
    '''Pad/truncate a count vector to length k.'''

    if len(x) >= k:
        return x[:k]
    return x + [0] * (k - len(x))

def _priorities(
    number_class: int,
    target_ratios: list[float],
    current_counts: list[int],
    eps: float
) -> list[float]:
    '''Helper: compute priority (normalized shortfall 0..1)'''

    p: list[float] = []
    for i in range(number_class):
        short = max(0.0, target_ratios[i] - current_counts[i])
        p.append(short / (target_ratios[i] + eps))
    return p
