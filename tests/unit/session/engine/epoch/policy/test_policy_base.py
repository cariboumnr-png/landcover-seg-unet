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

'''Unit tests for policy base module (policy/base.py).'''

# third-party imports
import torch
# local imports
import landseg.session.engine.epoch.policy.base as base_mod


# ----- `EngineBase` initialization and property delegate tests
def test_engine_base_init_and_properties(
    mock_runtime,
    mock_dataloaders,
    mock_dispatcher
):
    '''
    Given: Shared engine runtime, dataloaders, and dispatcher.
    When: Instantiating `EngineBase`.
    Then: Property delegates return runtime components correctly.
    '''
    engine_base = base_mod.EngineBase(
        engine_runtime=mock_runtime,
        dataloaders=mock_dataloaders,
        dispatcher=mock_dispatcher,
        device='cpu'
    )

    assert engine_base.engine is mock_runtime.engine
    assert engine_base.model is mock_runtime.engine.model
    assert engine_base.state is mock_runtime.engine.state
    assert engine_base.headspecs is mock_runtime.engine_tasks.headspecs
    assert engine_base.headlosses is mock_runtime.engine_tasks.headlosses
    assert engine_base.headmetrics is mock_runtime.engine_tasks.headmetrics
    assert engine_base.optimization is mock_runtime.engine_optim


# ----- `set_head_state` and `reset_head_state` tests
def test_engine_base_set_head_state_default(
    mock_runtime,
    mock_dataloaders,
    mock_dispatcher
):
    '''
    Given: `EngineBase` with active heads set to None.
    When: Invoking `set_head_state(active_heads=None)`.
    Then: Active heads default to `all_heads` and state pointers are updated.
    '''
    engine_base = base_mod.EngineBase(
        engine_runtime=mock_runtime,
        dataloaders=mock_dataloaders,
        dispatcher=mock_dispatcher,
        device='cpu'
    )

    engine_base.set_head_state(active_heads=None, frozen_heads=['head_2'])

    assert engine_base.state.heads.active_heads == ['head_1']
    assert engine_base.state.heads.frozen_heads == ['head_2']
    assert 'head_1' in engine_base.state.heads.active_hspecs
    assert 'head_1' in engine_base.state.heads.active_hloss
    assert 'head_1' in engine_base.state.heads.active_hmetrics
    assert engine_base.state.heads.multihead_metrics is None


def test_engine_base_reset_head_state(
    mock_runtime,
    mock_dataloaders,
    mock_dispatcher
):
    '''
    Given: `EngineBase` with populated active head state.
    When: Invoking `reset_head_state()`.
    Then: State pointers and model head flags are reset to None.
    '''
    engine_base = base_mod.EngineBase(
        engine_runtime=mock_runtime,
        dataloaders=mock_dataloaders,
        dispatcher=mock_dispatcher,
        device='cpu'
    )
    engine_base.set_head_state(active_heads=['head_1'])

    engine_base.reset_head_state()

    assert engine_base.state.heads.active_heads is None
    assert engine_base.state.heads.frozen_heads is None
    assert engine_base.state.heads.active_hspecs is None
    assert engine_base.state.heads.active_hloss is None
    assert engine_base.state.heads.active_hmetrics is None


# ----- `_batch_reset` batch helper tests
def test_engine_base_batch_reset(
    mock_runtime,
    mock_dataloaders,
    mock_dispatcher
):
    '''
    Given: `EngineBase` instance.
    When: Calling `_batch_reset` with new batch index and data tuple.
    Then: `batch_cxt` and `batch_out` in `EngineState` are refreshed.
    '''
    engine_base = base_mod.EngineBase(
        engine_runtime=mock_runtime,
        dataloaders=mock_dataloaders,
        dispatcher=mock_dispatcher,
        device='cpu'
    )
    batch = (torch.zeros(2, 3, 16, 16), torch.ones(2, 1, 16, 16), {})

    engine_base._batch_reset(bidx=5, _batch=batch)

    assert engine_base.state.batch_cxt.bidx == 5
    assert engine_base.state.batch_cxt.batch is batch
    assert engine_base.state.batch_out.bidx == 5
