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

'''Unit tests for state module (state.py).'''

# third-party imports
import torch
# local imports
import landseg.session.engine.runtime.executor.state as state_mod


# ----- `_Progress` dataclass tests
def test_progress_dataclass_defaults():
    '''
    Given: No arguments passed to `_Progress`.
    When: Instantiating `_Progress`.
    Then: `epoch` and `global_step` default to 0.
    '''
    progress = state_mod._Progress()

    assert progress.epoch == 0
    assert progress.global_step == 0


# ----- `_Heads` dataclass tests
def test_heads_dataclass_defaults():
    '''
    Given: No arguments passed to `_Heads`.
    When: Instantiating `_Heads`.
    Then: `all_heads` is empty list and pointers default to None.
    '''
    heads = state_mod._Heads()

    assert not heads.all_heads
    assert heads.active_heads is None
    assert heads.frozen_heads is None
    assert heads.active_hspecs is None
    assert heads.active_hloss is None
    assert heads.active_hmetrics is None
    assert heads.multihead_metrics is None
    assert heads.multihead_regularization is None


# ----- `_BatchContex` dataclass tests
def test_batch_context_defaults():
    '''
    Given: Default `_BatchContex` instantiation.
    When: Inspecting initial attributes.
    Then: Default placeholders and empty collections are present.
    '''
    ctx = state_mod._BatchContex()

    assert ctx.bidx == 0
    assert ctx.pidx_start == 0
    assert ctx.batch_size == 0
    assert ctx.batch is None
    assert ctx.x.numel() == 0
    assert not ctx.y_dict
    assert not ctx.domain


def test_batch_context_refresh():
    '''
    Given: Active `_BatchContex` populated with previous batch data.
    When: Calling `refresh` with new batch index and new batch tuple.
    Then: `bidx` and `pidx_start` are updated and collections cleared.
    '''
    ctx = state_mod._BatchContex(
        bidx=1,
        pidx_start=0,
        batch_size=16,
        x=torch.tensor([1.0]),
        y_dict={'head_1': torch.tensor([1])},
        domain={'ids': torch.tensor([0])}
    )
    new_batch = (torch.zeros(16, 3, 32, 32), torch.zeros(16, 1, 32, 32), {})

    ctx.refresh(bidx=3, batch=new_batch)

    assert ctx.bidx == 3
    assert ctx.batch is new_batch
    assert ctx.pidx_start == (3 - 1) * 16
    assert ctx.x.numel() == 0
    assert not ctx.y_dict
    assert not ctx.domain


# ----- `_BatchOutput` dataclass tests
def test_batch_output_defaults():
    '''
    Given: Default `_BatchOutput` instantiation.
    When: Inspecting initial attributes.
    Then: Default placeholders and empty dicts are set.
    '''
    out = state_mod._BatchOutput()

    assert out.bidx == 0
    assert not out.preds
    assert out.total_objective.numel() == 0
    assert not out.head_losses
    assert not out.regularization


def test_batch_output_refresh():
    '''
    Given: Populated `_BatchOutput` instance.
    When: Calling `refresh` with new batch index.
    Then: Outputs are cleared and `bidx` is updated.
    '''
    out = state_mod._BatchOutput(
        bidx=1,
        preds={'head_1': torch.tensor([0.5])},
        total_objective=torch.tensor(1.5),
        head_losses={'head_1': 1.5},
        regularization={'reg_1': 0.1}
    )

    out.refresh(bidx=2)

    assert out.bidx == 2
    assert not out.preds
    assert out.total_objective.numel() == 0
    assert not out.head_losses
    assert not out.regularization


# ----- `_InferOutput` dataclass tests
def test_infer_output_defaults_and_clear():
    '''
    Given: `_InferOutput` instance with spatial patch mappings.
    When: Calling `clear()`.
    Then: All spatial grid patch dictionaries are emptied.
    '''
    infer = state_mod._InferOutput(
        inputs={(0, 0): torch.zeros(3, 32, 32)},
        labels={'head_1': {(0, 0): torch.zeros(32, 32)}},
        preds={'head_1': {(0, 0): torch.zeros(32, 32)}},
        errors={'head_1': {(0, 0): torch.zeros(32, 32)}}
    )

    assert len(infer.inputs) == 1
    infer.clear()

    assert not infer.inputs
    assert not infer.labels
    assert not infer.preds
    assert not infer.errors


# ----- `_OptimRuntime` dataclass tests
def test_optim_runtime_lr_property():
    '''
    Given: `_OptimRuntime` with non-empty learning rates list.
    When: Reading `lr` property.
    Then: Return the first element of `lrs`, or `None` if empty.
    '''
    scaler = torch.GradScaler(device='cpu', enabled=False)
    opt_runtime_empty = state_mod._OptimRuntime(scaler=scaler, lrs=[])
    opt_runtime_active = state_mod._OptimRuntime(scaler=scaler, lrs=[1e-3, 1e-4])

    assert opt_runtime_empty.lr is None
    assert opt_runtime_active.lr == 1e-3


# ----- `initialize_state` public function tests
def test_initialize_state_creation():
    '''
    Given: Target parameters for `initialize_state`.
    When: Invoking `initialize_state`.
    Then: Return fully configured `EngineState` with active AMP scaler.
    '''
    all_heads = ['head_1', 'head_2']
    state = state_mod.initialize_state(
        all_heads=all_heads,
        batch_size=8,
        use_amp=False,
        device='cpu'
    )

    assert isinstance(state, state_mod.EngineState)
    assert state.heads.all_heads == all_heads
    assert state.batch_cxt.batch_size == 8
    assert state.optim.scaler is not None
