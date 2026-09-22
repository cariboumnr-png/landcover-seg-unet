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
Diagnostic metrics for multi-head and multi-task learning models.

Provides horizontal evaluation of model performance across independent
and hierarchical tasks, including Global Exact Match (GEM) and
logical constraint violation detection.

Public APIs:
    - `MTLMetricsAggregator`: Cross-head aggregator for GEM and
      constraint violation metrics.
'''

# standard imports
import dataclasses
import typing
# third-party imports
import torch
# local imports
import landseg.session.contracts as contracts
import landseg.session.engine.tasks.constraints as constraints


# ----- private dataclasses
@dataclasses.dataclass
class _Tally:
    '''Internal counter for horizontal metrics.'''

    hits: int = 0
    samples: int = 0


# ----- public classes
class MTLMetricsAggregator:
    '''
    Aggregator for multi-task learning metrics across multiple heads.

    Calculates:
    1. Global Exact Match (GEM): Per-pixel accuracy across all heads.
    2. Constraint Violations: Inconsistencies between predicted classes.
    '''

    def __init__(
        self,
        cons: typing.Sequence[constraints.MTLConstraint] | None,
        *,
        ignore_index: int,
    ):
        '''
        Initialize the aggregator.

        Args:
            cons:
                optional list of logical constraints to evaluate.
            ignore_index:
                index to ignore in ground truth for GEM and validity
                masks.
        '''
        self.ignore_index = ignore_index
        self.constraints = cons or []

        # internal counters
        self.gem_hits: int = 0
        self.gem_samples: int = 0

        # violation counters: {constraint_name: Tally}
        self.violations: dict[str, _Tally] = {
            c.name: _Tally() for c in self.constraints
        }

    @torch.no_grad()
    def update(
        self,
        preds_1b: contracts.TensorDict,
        targets_1b: contracts.TensorDict
    ) -> None:
        '''
        Update global metrics with predictions and targets for a batch.

        Args:
            preds_1b:
                predicted class IDs (1-based) per head.
            targets_1b:
                ground truth labels (1-based) per head.
        '''
        if not preds_1b or not targets_1b:
            return
        common_heads = [h for h in targets_1b if h in preds_1b]
        if not common_heads:
            return

        self._get_gem(preds_1b, targets_1b, common_heads)
        self._check_violations(preds_1b, targets_1b)

    def compute(self) -> dict[str, float]:
        '''Compute final ratios for GEM and violations.'''
        results = {}
        if self.gem_samples > 0:
            results['gem'] = float(self.gem_hits) / self.gem_samples
        for name, tally in self.violations.items():
            if tally.samples > 0:
                results[f'violation_{name}'] = float(tally.hits) / tally.samples
        return results

    def reset(self) -> None:
        '''Zero all counters and move state to device if needed.'''
        self.gem_hits = 0
        self.gem_samples = 0
        for tally in self.violations.values():
            tally.hits = 0
            tally.samples = 0

    def _get_gem(
        self,
        preds_1b: contracts.TensorDict,
        targets_1b: contracts.TensorDict,
        common_heads: list[str]
    ):
        '''Check GEM logic.'''
        first_head = common_heads[0]
        joint_valid = torch.ones_like(targets_1b[first_head], dtype=torch.bool)
        joint_match = torch.ones_like(targets_1b[first_head], dtype=torch.bool)
        for head in common_heads:
            t, p = targets_1b[head], preds_1b[head]
            joint_valid &= (t != self.ignore_index)
            joint_match &= (p == t)

        self.gem_hits += int((joint_match & joint_valid).sum().item())
        self.gem_samples += int(joint_valid.sum().item())

    def _check_violations(
        self,
        preds_1b: contracts.TensorDict,
        targets_1b: contracts.TensorDict,
    ):
        '''Check violations.'''
        for c in self.constraints:
            if c.source_head not in preds_1b or c.target_head not in preds_1b:
                continue

            p_src, p_tgt = preds_1b[c.source_head], preds_1b[c.target_head]
            t_src, t_tgt = targets_1b[c.source_head], targets_1b[c.target_head]

            valid_mask = (
                (t_src != self.ignore_index) &
                (t_tgt != self.ignore_index)
            )
            forbidden_tensor = torch.tensor(c.forbidden, device=p_tgt.device)
            violation_mask = (
                (p_src == c.trigger_val) &
                torch.isin(p_tgt, forbidden_tensor)
            )

            tally = self.violations[c.name]
            tally.hits += int((violation_mask & valid_mask).sum().item())
            tally.samples += int(valid_mask.sum().item())
