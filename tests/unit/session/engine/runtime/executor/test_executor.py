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

# pylint: disable=protected-access

'''Unit tests for executor module (executor.py and __init__.py).'''

# standard imports
import dataclasses
# third-party imports
import pytest
import torch
# local imports
import landseg.session.engine.runtime.executor.executor as exec_mod
import landseg.session.engine.runtime.executor.state as state_mod


# ----- `BatchExecContext` dataclass tests
def test_batch_exec_context_instantiation(dummy_exec_context):
    '''
    Given: Parameters for `BatchExecContext`.
    When: Instantiating `BatchExecContext`.
    Then: Attributes are assigned correctly.
    '''
    assert dummy_exec_context.device == 'cpu'
    assert dummy_exec_context.patch_per_blk == 4


# ----- `BatchEngine` initialization and batch parsing tests
def test_batch_engine_init(
    dummy_multihead_model,
    dummy_exec_context,
    session_config
):
    '''
    Given: Multihead model, state, config, and context.
    When: Instantiating `BatchEngine`.
    Then: Model is configured with logit adjustment alpha and assigned device.
    '''
    state = state_mod.initialize_state(
        all_heads=['head_1'],
        batch_size=2,
        use_amp=False,
        device='cpu'
    )
    config = dataclasses.replace(
        session_config.engine_exec,
        logit_adjust_alpha=0.8
    )

    engine = exec_mod.BatchEngine(
        model=dummy_multihead_model,
        engine_state=state,
        config=config,
        context=dummy_exec_context
    )

    assert engine.model.logit_adjust_alpha == 0.8
    assert engine.device == 'cpu'


def test_batch_engine_parse_batch_labeled(
    dummy_multihead_model,
    dummy_exec_context,
    session_config
):
    '''
    Given: Batch engine with a labeled training batch (x, y, domain).
    When: Executing `_parse_batch`.
    Then: `batch_cxt` receives parsed tensors `x`, `y_dict`, and `domain`.
    '''
    state = state_mod.initialize_state(
        all_heads=['head_1'],
        batch_size=2,
        use_amp=False,
        device='cpu'
    )
    x = torch.randn(2, 3, 16, 16)
    y = torch.ones(2, 1, 16, 16, dtype=torch.long)
    domain = {'ids': torch.tensor([0, 1])}

    state.batch_cxt.refresh(bidx=1, batch=(x, y, domain))

    engine = exec_mod.BatchEngine(
        model=dummy_multihead_model,
        engine_state=state,
        config=session_config.engine_exec,
        context=dummy_exec_context
    )
    engine._parse_batch()

    assert engine.state.batch_cxt.x.shape == (2, 3, 16, 16)
    assert 'head_1' in engine.state.batch_cxt.y_dict
    assert engine.state.batch_cxt.y_dict['head_1'].shape == (2, 16, 16)
    assert 'ids_domain' in engine.state.batch_cxt.domain


def test_batch_engine_parse_batch_invalid_active_head_raises(
    dummy_multihead_model,
    dummy_exec_context,
    session_config
):
    '''
    Given: State with an `active_heads` list containing unknown heads.
    When: Executing `_parse_batch`.
    Then: Raise `KeyError`.
    '''
    state = state_mod.initialize_state(
        all_heads=['head_1'],
        batch_size=2,
        use_amp=False,
        device='cpu'
    )
    state.heads.active_heads = ['head_unknown']
    x = torch.randn(2, 3, 16, 16)
    y = torch.ones(2, 1, 16, 16, dtype=torch.long)

    state.batch_cxt.refresh(bidx=1, batch=(x, y, {}))

    engine = exec_mod.BatchEngine(
        model=dummy_multihead_model,
        engine_state=state,
        config=session_config.engine_exec,
        context=dummy_exec_context
    )

    with pytest.raises(KeyError, match='Active heads not found'):
        engine._parse_batch()


# ----- `run_train_batch` execution tests
def test_run_train_batch(
    dummy_multihead_model,
    dummy_head_loss,
    dummy_head_spec,
    dummy_exec_context,
    session_config
):
    '''
    Given: Configured `BatchEngine` with loss modules.
    When: Invoking `run_train_batch`.
    Then: Complete forward pass and write objective outputs into state.
    '''
    state = state_mod.initialize_state(
        all_heads=['head_1'],
        batch_size=2,
        use_amp=False,
        device='cpu'
    )
    state.heads.active_hspecs = {'head_1': dummy_head_spec}
    state.heads.active_hloss = {'head_1': dummy_head_loss}

    x = torch.randn(2, 3, 16, 16)
    y = torch.ones(2, 1, 16, 16, dtype=torch.long)
    state.batch_cxt.refresh(bidx=1, batch=(x, y, {}))

    engine = exec_mod.BatchEngine(
        model=dummy_multihead_model,
        engine_state=state,
        config=session_config.engine_exec,
        context=dummy_exec_context
    )

    engine.run_train_batch()

    assert 'head_1' in engine.state.batch_out.preds
    assert engine.state.batch_out.total_objective.numel() == 1
    assert 'head_1' in engine.state.batch_out.head_losses


# ----- `run_validate_batch` execution tests
def test_run_validate_batch(
    dummy_multihead_model,
    dummy_head_metric,
    dummy_exec_context,
    session_config
):
    '''
    Given: Configured `BatchEngine` with metric accumulators.
    When: Invoking `run_validate_batch`.
    Then: Complete validation forward pass and update metrics.
    '''
    state = state_mod.initialize_state(
        all_heads=['head_1'],
        batch_size=2,
        use_amp=False,
        device='cpu'
    )
    state.heads.active_hmetrics = {'head_1': dummy_head_metric}

    x = torch.randn(2, 3, 16, 16)
    y = torch.ones(2, 1, 16, 16, dtype=torch.long)
    state.batch_cxt.refresh(bidx=1, batch=(x, y, {}))

    engine = exec_mod.BatchEngine(
        model=dummy_multihead_model,
        engine_state=state,
        config=session_config.engine_exec,
        context=dummy_exec_context
    )

    engine.run_validate_batch()

    assert 'head_1' in engine.state.batch_out.preds
    assert dummy_head_metric.updated is True


# ----- `run_infer_batch` and spatial aggregation tests
def test_run_infer_batch_spatial_aggregation(
    dummy_multihead_model,
    dummy_head_metric,
    dummy_exec_context,
    session_config
):
    '''
    Given: Inference batch with patch spatial layout details.
    When: Invoking `run_infer_batch`.
    Then: Aggregate predictions into spatial grid coordinates in `infer_out`.
    '''
    state = state_mod.initialize_state(
        all_heads=['head_1'],
        batch_size=1,
        use_amp=False,
        device='cpu'
    )
    state.heads.active_hmetrics = {'head_1': dummy_head_metric}

    x = torch.randn(1, 3, 16, 16)
    y = torch.ones(1, 1, 16, 16, dtype=torch.long)
    state.batch_cxt.refresh(bidx=1, batch=(x, y, {}))

    engine = exec_mod.BatchEngine(
        model=dummy_multihead_model,
        engine_state=state,
        config=session_config.engine_exec,
        context=dummy_exec_context
    )

    engine.run_infer_batch()

    assert (0, 0) in engine.state.infer_out.inputs
    assert (0, 0) in engine.state.infer_out.preds['head_1']
    assert (0, 0) in engine.state.infer_out.labels['head_1']
