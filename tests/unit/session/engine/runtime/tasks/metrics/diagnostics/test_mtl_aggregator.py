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

'''Unit tests for MTL metrics aggregator module (mtl_aggregator.py).'''

# standard imports
import dataclasses
# third-party imports
import torch
# local imports
import landseg.session.engine.runtime.tasks.metrics.diagnostics.mtl_aggregator as aggregator_module


@dataclasses.dataclass
class DummyConstraint:
    '''Dummy constraint conforming to `MTLConstraint` protocol.'''
    name: str
    source_head: str
    trigger_val: int
    target_head: str
    forbidden: list[int]


def test_mtl_aggregator_init():
    '''
    Given: Optional list of constraints and ignore index.
    When: Instantiating `MTLMetricsAggregator`.
    Then: Initialize internal counters and violation tallies correctly.
    '''
    cons = [
        DummyConstraint(
            name='rule1',
            source_head='head_a',
            trigger_val=1,
            target_head='head_b',
            forbidden=[2]
        )
    ]
    agg = aggregator_module.MTLMetricsAggregator(cons, ignore_index=255)

    assert agg.ignore_index == 255
    assert len(agg.constraints) == 1
    assert agg.gem_hits == 0
    assert agg.gem_samples == 0
    assert 'rule1' in agg.violations
    assert agg.violations['rule1'].hits == 0
    assert agg.violations['rule1'].samples == 0


def test_mtl_aggregator_update_empty_or_disjoint():
    '''
    Given: Empty inputs or inputs without common prediction heads.
    When: Calling `update`.
    Then: Perform early exit without updating counters.
    '''
    agg = aggregator_module.MTLMetricsAggregator([], ignore_index=255)

    # empty dicts
    agg.update({}, {})
    assert agg.gem_samples == 0

    # disjoint heads
    preds = {'head_a': torch.tensor([[1]])}
    targets = {'head_b': torch.tensor([[1]])}
    agg.update(preds, targets)
    assert agg.gem_samples == 0


def test_mtl_aggregator_gem_calculation():
    '''
    Given: Ground truth targets and predictions across common heads.
    When: Calling `update`.
    Then: Correctly calculate global exact match hits and valid samples.
    '''
    agg = aggregator_module.MTLMetricsAggregator([], ignore_index=255)

    # 1x2 spatial predictions and targets for 2 heads
    # pixel (0,0): match in both heads (1 == 1, 2 == 2) -> GEM hit
    # pixel (0,1): match in head_a (1 == 1) but mismatch in head_b (2 != 1) -> not GEM hit
    preds = {
        'head_a': torch.tensor([[1, 1]]),
        'head_b': torch.tensor([[2, 2]])
    }
    targets = {
        'head_a': torch.tensor([[1, 1]]),
        'head_b': torch.tensor([[2, 1]])
    }

    agg.update(preds, targets)

    assert agg.gem_hits == 1
    assert agg.gem_samples == 2

    # test ignore_index filtering in GEM
    # pixel (0,1) ignored in head_a -> total valid samples reduced to 1
    targets_ignore = {
        'head_a': torch.tensor([[1, 255]]),
        'head_b': torch.tensor([[2, 1]])
    }
    agg.reset()
    agg.update(preds, targets_ignore)

    assert agg.gem_hits == 1
    assert agg.gem_samples == 1


def test_mtl_aggregator_constraint_violations():
    '''
    Given: Predictions triggering a logical constraint violation.
    When: Calling `update`.
    Then: Correctly tally constraint violation hits and valid samples.
    '''
    cons = [
        DummyConstraint(
            name='c1',
            source_head='head_a',
            trigger_val=1,
            target_head='head_b',
            forbidden=[2, 3]
        )
    ]
    agg = aggregator_module.MTLMetricsAggregator(cons, ignore_index=255)

    # pixel (0,0): source=1 (triggered), target=2 (forbidden) -> violation!
    # pixel (0,1): source=1 (triggered), target=1 (allowed) -> no violation
    # pixel (0,2): source=2 (not triggered), target=2 (forbidden) -> no violation
    preds = {
        'head_a': torch.tensor([[1, 1, 2]]),
        'head_b': torch.tensor([[2, 1, 2]])
    }
    targets = {
        'head_a': torch.tensor([[1, 1, 1]]),
        'head_b': torch.tensor([[1, 1, 1]])
    }

    agg.update(preds, targets)

    assert agg.violations['c1'].hits == 1
    assert agg.violations['c1'].samples == 3


def test_mtl_aggregator_compute_and_reset():
    '''
    Given: An aggregator populated with update samples.
    When: Calling `compute` and then `reset`.
    Then: Return calculated metrics dict and reset all counters to zero.
    '''
    cons = [
        DummyConstraint(
            name='c1',
            source_head='head_a',
            trigger_val=1,
            target_head='head_b',
            forbidden=[2]
        )
    ]
    agg = aggregator_module.MTLMetricsAggregator(cons, ignore_index=255)

    preds = {
        'head_a': torch.tensor([[1, 1]]),
        'head_b': torch.tensor([[2, 1]])
    }
    targets = {
        'head_a': torch.tensor([[1, 1]]),
        'head_b': torch.tensor([[2, 1]])
    }

    agg.update(preds, targets)
    metrics_res = agg.compute()

    assert 'gem' in metrics_res
    assert metrics_res['gem'] == 1.0
    assert 'violation_c1' in metrics_res
    assert metrics_res['violation_c1'] == 0.5

    agg.reset()
    assert agg.gem_hits == 0
    assert agg.gem_samples == 0
    assert agg.violations['c1'].hits == 0
    assert agg.violations['c1'].samples == 0
    results = agg.compute()
    assert isinstance(results, dict) and len(results) == 0


def test_mtl_aggregator_constraint_skipped_missing_head():
    '''
    Given: A constraint referencing a head not present in `preds_1b`.
    When: Calling `update`.
    Then: Skip constraint check without raising error.
    '''
    cons = [
        DummyConstraint(
            name='c1',
            source_head='head_a',
            trigger_val=1,
            target_head='head_c',  # head_c not in preds_1b
            forbidden=[2]
        )
    ]
    agg = aggregator_module.MTLMetricsAggregator(cons, ignore_index=255)

    preds = {
        'head_a': torch.tensor([[1, 1]]),
        'head_b': torch.tensor([[2, 1]])
    }
    targets = {
        'head_a': torch.tensor([[1, 1]]),
        'head_b': torch.tensor([[2, 1]])
    }

    agg.update(preds, targets)

    assert agg.violations['c1'].hits == 0
    assert agg.violations['c1'].samples == 0
