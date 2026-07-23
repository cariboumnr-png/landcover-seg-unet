# =========================================================================== #
#           Copyright (c) His Majesty the King in right of Ontario,           #
#         as represented by the Minister of Natural Resources, 2026.          #
#                                                                             #
#                      (c) King's Printer for Ontario, 2026.                  #
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

# pylint: disable=duplicate-code
# pylint: disable=protected-access
# pylint: disable=redefined-outer-name

'''Unit tests for evaluator policy module (policy/evaluator.py).'''

# third-party imports
import torch
# local imports
import landseg.core as core
import landseg.session.engine.epoch.policy.evaluator as eval_mod


# ----- `MultiHeadEvaluator` initialization tests
def test_evaluator_init(mock_runtime, mock_dataloaders, mock_dispatcher):
    '''
    Given: Parameters for `MultiHeadEvaluator`.
    When: Instantiating `MultiHeadEvaluator`.
    Then: Validation and inference intervals and results containers are set.
    '''
    evaluator = eval_mod.MultiHeadEvaluator(
        val_every=2,
        infer_every=3,
        dataset='val',
        engine_runtime=mock_runtime,
        dataloaders=mock_dataloaders,
        dispatcher=mock_dispatcher,
        device='cpu'
    )

    assert evaluator.val_every == 2
    assert evaluator.infer_every == 3
    assert isinstance(evaluator.val_results, core.ValStepResults)
    assert isinstance(evaluator.infer_results, core.InferStepResults)


# ----- `validate` execution tests
def test_evaluator_validate_interval_skip(
    mock_runtime,
    mock_dataloaders,
    mock_dispatcher
):
    '''
    Given: `MultiHeadEvaluator` with `val_every=2`.
    When: Calling `validate` with epoch=1.
    Then: Return None without executing validation loop.
    '''
    evaluator = eval_mod.MultiHeadEvaluator(
        val_every=2,
        engine_runtime=mock_runtime,
        dataloaders=mock_dataloaders,
        dispatcher=mock_dispatcher,
        device='cpu'
    )

    result = evaluator.validate(epoch=1)

    assert result is None
    assert 'on_val_policy_begin' not in mock_dispatcher.events


def test_evaluator_validate(
    mock_runtime,
    mock_dataloaders,
    mock_dispatcher
):
    '''
    Given: `MultiHeadEvaluator` configured with active heads.
    When: Calling `validate` on valid epoch boundary (epoch=2).
    Then: Run validation, compute metrics, and return `ValStepResults`.
    '''
    evaluator = eval_mod.MultiHeadEvaluator(
        val_every=2,
        engine_runtime=mock_runtime,
        dataloaders=mock_dataloaders,
        dispatcher=mock_dispatcher,
        device='cpu'
    )
    evaluator.set_head_state(active_heads=['head_1'])

    results = evaluator.validate(epoch=2)

    assert isinstance(results, core.ValStepResults)
    assert 'head_1' in results.head_metrics
    assert 'on_val_policy_begin' in mock_dispatcher.events
    assert 'on_val_policy_end' in mock_dispatcher.events


# ----- `infer` and patch stitching execution tests
def test_evaluator_infer(
    mock_runtime,
    mock_dataloaders,
    mock_dispatcher
):
    '''
    Given: `MultiHeadEvaluator` with active heads and spatial patch context.
    When: Calling `infer` on valid epoch boundary.
    Then: Run continuous inference, stitch patches, and return `InferStepResults`.
    '''
    evaluator = eval_mod.MultiHeadEvaluator(
        infer_every=1,
        engine_runtime=mock_runtime,
        dataloaders=mock_dataloaders,
        dispatcher=mock_dispatcher,
        device='cpu'
    )
    evaluator.set_head_state(active_heads=['head_1'])

    # populate dummy patch entries in infer_out for stitching
    coord = (0, 0)
    evaluator.state.infer_out.labels['head_1'] = {coord: torch.zeros(16, 16)}
    evaluator.state.infer_out.preds['head_1'] = {coord: torch.zeros(16, 16)}
    evaluator.state.infer_out.errors['head_1'] = {coord: torch.zeros(16, 16)}

    results = evaluator.infer(epoch=1)

    assert isinstance(results, core.InferStepResults)
    assert 'head_1' in results.head_metrics
    assert 'head_1' in results.infer_preds
    assert 'on_infer_policy_begin' in mock_dispatcher.events
    assert 'on_infer_policy_end' in mock_dispatcher.events
