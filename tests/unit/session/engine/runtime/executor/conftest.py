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

# pylint: disable=missing-function-docstring
# pylint: disable=protected-access
# pylint: disable=redefined-outer-name
# pylint: disable=too-few-public-methods

'''Fixtures for testing `landseg.session.engine.runtime.executor` module.'''

# third-party imports
import pytest
import torch
# local imports
import landseg.session.engine.runtime.executor.executor as exec_mod
import landseg.session.engine.runtime.tasks.heads.specs as specs_mod


class DummyMultiheadModel(torch.nn.Module):
    '''Dummy multihead model implementing `MultiheadModelLike`.'''
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 2, kernel_size=1)
        self.logit_adjust_alpha = 1.0

    def set_logit_adjust_alpha(self, alpha: float) -> None:
        self.logit_adjust_alpha = alpha

    def forward(self, x, ids_domain=None, vec_domain=None):
        _ = ids_domain, vec_domain
        out = self.conv(x)
        return {'head_1': out}


class DummyHeadLoss(torch.nn.Module):
    '''Dummy composite head loss module.'''
    def __init__(self, ignore_index: int = 0):
        super().__init__()
        self.ignore_index = ignore_index

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        masks: dict[float, torch.Tensor] | None = None,
        features: torch.Tensor | None = None
    ) -> torch.Tensor:
        _ = masks, features
        return (pred - target).pow(2).mean()


class DummyHeadMetric:
    '''Dummy head confusion matrix metric accumulator.'''
    def __init__(self):
        self.updated = False

    def update(self, logits, targets, parent_raw_1b=None):
        _ = logits, targets, parent_raw_1b
        self.updated = True


class DummyRegularizer(torch.nn.Module):
    '''Dummy consistency regularizer for multihead testing.'''
    def __init__(self, reduction: str = 'mean', val: float = 0.5):
        super().__init__()
        self.reduction = reduction
        self.val = val

    def forward(
        self,
        multihead_preds: dict[str, torch.Tensor],
        multihead_targets: dict[str, torch.Tensor]
    ) -> torch.Tensor:
        _ = multihead_preds, multihead_targets
        return torch.tensor(self.val)


@pytest.fixture
def dummy_multihead_model():
    return DummyMultiheadModel()


@pytest.fixture
def dummy_head_loss():
    return DummyHeadLoss()


@pytest.fixture
def dummy_head_metric():
    return DummyHeadMetric()


@pytest.fixture
def dummy_regularizer():
    return DummyRegularizer()


@pytest.fixture
def dummy_head_spec():
    return specs_mod.HeadSpec(
        name='head_1',
        count=[10, 10],
        loss_alpha=[0.5, 0.5],
        parent_head=None,
        parent_cls=None,
        weight=1.0,
        exclude_cls=None
    )


@pytest.fixture
def dummy_exec_context():
    return exec_mod.BatchExecContext(
        parent_map={},
        patch_per_blk=4,
        patch_per_dim=2,
        block_columns=2,
        device='cpu'
    )
