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
Validation dataset scoring utilities that rank blocks by their label-
distribution similarity to a target. Computes target distributions from
global counts and scores blocks with a weighted L1 metric and reward
terms.

Public APIs:
    - BlockScore: Dataclass representing a scored block and its metadata.
    - ScoreParams: Dataclass holding scoring hyperparameters.
    - score_blocks: Compute and persist per-block scores based on label
      distributions.
'''

# standard imports
import dataclasses
import math
import os
import typing
# third-party imports
import numpy
# local imports
import landseg.utils as utils

# ------------------------------Public  Dataclass------------------------------
@dataclasses.dataclass
class ScoringConfig:
    '''Score configuration.'''
    alpha: float                    # exponent for transforming block counts
    beta: float                     # reward weight for classes during L1
    epsilon: float                  # small constant for numerical stability
    reward: tuple[int, ...]         # class indices to reward (0-based)

# -------------------------------Public Function-------------------------------
def score_blocks(
    global_class_count: list[int],
    input_blocks: dict[str, list[int]],
    config: ScoringConfig,
    scores_path: str,
    *,
    rescore: bool = False,
    **kwargs
) -> None:
    '''
    Score input blocks based on label class distributions.

    Args:
        global_cls_count: Global per-class counts by head used to derive
            the target distribution.
        input_blocks: Mapping from block names to file paths to score.
        params: Scoring parameters (head/alpha/beta/eps/reward classes).
        scores_path: Output JSON path for persisted, sorted scores.
        rescore: If True, recompute even if scores_path already exists.
    '''

    # read saved scores if already exist
    if os.path.exists(scores_path) and not rescore:
        return

    # target inverse probability (p) from global class distribution
    p = _count_to_prob_w_temp(global_class_count, 1.0, config.epsilon)
    # scoring mode as kwargs
    kws = {'mode': kwargs.get('mode', 'log_lift')}
    # score blocks from list with parallel processing
    jobs = [(_score, (k, v, p, config), kws) for k, v in input_blocks.items()]
    scores: list[tuple[str, float]] = utils.ParallelExecutor().run(jobs)
    # sort blocks to rank in descending order
    sorted_scores = sorted(scores, key=lambda x: x[1] or -1e9, reverse=True)

    # save sorted scores to a file
    utils.write_json(scores_path, dict(sorted_scores))
    utils.hash_artifacts(scores_path)

# ------------------------------private  function------------------------------
def _score(
    blk_name: str,
    blk_count: list[int],
    target_p: numpy.ndarray,
    config: ScoringConfig,
    *,
    mode: typing.Literal['log_lift', 'weighted_l1']
) -> tuple[str, float | None]:
    '''Score a single block against the target distribution.'''

    # log lift on reward classes
    if mode == 'log_lift':
        block_p = _count_to_prob_w_temp(blk_count, config.alpha, config.epsilon)
        _score = _log_lift_on_reward(
            target_p,
            block_p,
            config.reward,
            eps=config.epsilon
        )
    # L1 distance to measure pairwise closeness
    elif mode == 'weighted_l1':
        block_p = _count_to_prob_w_temp(blk_count, config.alpha, config.epsilon)
        _score = _weighted_l1_w_reward(
            target_p,
            block_p,
            config.reward,
            beta=config.beta,
            eps=config.epsilon
        )
    else:
        raise ValueError(f'Invalid scoring mode: {mode}')
    # return
    return blk_name, _score

def _count_to_prob_w_temp(
    cls_counts: list[int],
    alpha: float,
    eps: float
) -> numpy.ndarray:
    '''Counts to a smoothed prob. distrib. with temperature scaling.'''

    # safe count to probability distribution
    arr = numpy.asarray(cls_counts)
    total = arr.sum()
    if total == 0.0:
        return numpy.zeros_like(cls_counts)
    # re-normalize to have probability sum==1
    p = arr / total
    p = numpy.clip(p, eps, None) # avoid downstream log(0)
    p = p / p.sum()
    # inverse and return
    if alpha != 1.0:
        p = p ** alpha
        p /= p.sum()
    return p

def _log_lift_on_reward(
    p: numpy.ndarray[tuple[int], numpy.dtype[numpy.float64]],
    q: numpy.ndarray[tuple[int], numpy.dtype[numpy.float64]],
    reward_cls: tuple[int, ...],
    *,
    eps: float
) -> float | None:
    '''Mass-weighted log-lift.'''

    # p as target q as test
    # no-ops if q is all zeros
    if sum(q) == 0:
        return None
    return sum(q[i] * math.log((q[i] + eps) / (p[i] + eps)) for i in reward_cls)

# now legacy
def _weighted_l1_w_reward(
    p: numpy.ndarray[tuple[int], numpy.dtype[numpy.float64]],
    q: numpy.ndarray[tuple[int], numpy.dtype[numpy.float64]],
    reward_cls: tuple[int, ...],
    *,
    beta: float,
    eps: float
) -> float:
    '''Weighted L1 distance with bonus on specified reward classes.'''

    # sanity checks
    assert numpy.isclose(p.sum(), 1), p.sum()
    assert numpy.isclose(q.sum(), 1), q.sum()
    # p as target q as test
    # use log weight and bonus for reward classes > target classes
    w_l1 = sum(abs(a - b) * (1 + abs(math.log(a + eps))) for a, b in zip(p, q))
    bonus = sum(max(0, q[i] - p[i]) for i in reward_cls) * beta
    return float(w_l1 - bonus)
