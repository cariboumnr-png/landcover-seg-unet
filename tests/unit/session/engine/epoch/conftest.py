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

# pylint: disable=missing-class-docstring
# pylint: disable=missing-function-docstring
# pylint: disable=protected-access
# pylint: disable=redefined-outer-name
# pylint: disable=too-few-public-methods

'''Fixtures for testing `landseg.session.engine.epoch` subpackage.'''

# standard imports
import dataclasses
# third-party imports
import pytest
import torch
# local imports
import landseg.session.engine.runtime.executor as executor_mod
import landseg.session.engine.runtime.executor.state as state_mod
import landseg.session.engine.runtime.optim.builder as optim_builder
import landseg.session.engine.runtime.tasks.heads.specs as specs_mod


# ----- mock helper classes
class MockMultiheadModel(torch.nn.Module):
    '''Mock multihead model for engine epoch tests.'''
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 2, kernel_size=1)
        self.active_heads: list[str] | None = None
        self.frozen_heads: list[str] | None = None
        self.logit_adjust_alpha: float = 1.0

    def set_active_heads(self, active_heads: list[str] | None) -> None:
        self.active_heads = active_heads

    def set_frozen_heads(self, frozen_heads: list[str] | None) -> None:
        self.frozen_heads = frozen_heads

    def reset_heads(self) -> None:
        self.active_heads = None
        self.frozen_heads = None

    def set_logit_adjust_alpha(self, alpha: float) -> None:
        self.logit_adjust_alpha = alpha

    def forward(self, x, ids_domain=None, vec_domain=None):
        _ = ids_domain, vec_domain
        return {'head_1': self.conv(x)}


class MockHeadLoss(torch.nn.Module):
    '''Mock composite loss module.'''
    def __init__(self, ignore_index: int = 0):
        super().__init__()
        self.ignore_index = ignore_index

    def forward(self, pred, target, masks=None, features=None):
        _ = masks, features
        return (pred - target).pow(2).mean()


class MockHeadMetric:
    '''Mock confusion matrix accumulator metric.'''
    def __init__(self):
        self.reset_called = False
        self.updated_called = False
        self.computed_called = False

    def reset(self, device: str | torch.device):
        _ = device
        self.reset_called = True

    def update(self, logits, targets, parent_raw_1b=None):
        _ = logits, targets, parent_raw_1b
        self.updated_called = True

    def compute(self) -> dict[str, float]:
        self.computed_called = True
        return {'iou': 0.85, 'accuracy': 0.90}


class MockPreviewContext:
    '''Mock preview context for continuous inference.'''
    patch_grid_shape: tuple[int, int] = (2, 2)


class MockDataloaderMeta:
    '''Mock dataloader metadata container.'''
    batch_size: int = 2
    patch_size: int = 16
    preview_context = MockPreviewContext()


class MockDataLoader:
    '''Mock dataloader yielding a fixed sequence of batches.'''
    def __init__(self, num_batches: int = 2):
        x = torch.randn(2, 3, 16, 16)
        y = torch.ones(2, 1, 16, 16, dtype=torch.long)
        self.batches = [(x, y, {}) for _ in range(num_batches)]

    def __len__(self) -> int:
        return len(self.batches)

    def __iter__(self):
        return iter(self.batches)


@dataclasses.dataclass
class MockDataLoaders:
    '''Mock dataloaders container matching `DataLoadersLike`.'''
    train: MockDataLoader | None = dataclasses.field(
        default_factory=lambda: MockDataLoader(2)
    )
    val: MockDataLoader | None = dataclasses.field(
        default_factory=lambda: MockDataLoader(2)
    )
    test: MockDataLoader | None = dataclasses.field(
        default_factory=lambda: MockDataLoader(2)
    )
    meta: MockDataloaderMeta = dataclasses.field(
        default_factory=MockDataloaderMeta
    )


class MockDispatcher:
    '''Mock session dispatcher tracking lifecycle callback calls.'''
    def __init__(self):
        self.events: list[str] = []

    def on_train_policy_begin(self):
        self.events.append('on_train_policy_begin')

    def on_train_policy_end(self, results):
        _ = results
        self.events.append('on_train_policy_end')

    def on_val_policy_begin(self):
        self.events.append('on_val_policy_begin')

    def on_val_policy_end(self, results):
        _ = results
        self.events.append('on_val_policy_end')

    def on_infer_policy_begin(self):
        self.events.append('on_infer_policy_begin')

    def on_infer_policy_end(self, results):
        _ = results
        self.events.append('on_infer_policy_end')

    def on_batch_begin(self, mode: str, bidx: int):
        _ = mode, bidx
        self.events.append('on_batch_begin')

    def on_train_batch_end(self, bidx: int, results):
        _ = bidx, results
        self.events.append('on_train_batch_end')

    def on_val_batch_end(self):
        self.events.append('on_val_batch_end')

    def on_infer_batch_end(self):
        self.events.append('on_infer_batch_end')


class MockEngineTasks:
    '''Mock tasks bundle containing per-head components.'''
    def __init__(self):
        self.headspecs = {
            'head_1': specs_mod.HeadSpec(
                name='head_1',
                count=[10, 10],
                loss_alpha=[0.5, 0.5],
                parent_head=None,
                parent_cls=None,
                weight=1.0,
                exclude_cls=None
            )
        }
        self.headlosses = {'head_1': MockHeadLoss()}
        self.headmetrics = {'head_1': MockHeadMetric()}
        self.multihead_metrics = None
        self.multihead_regularization = None


@dataclasses.dataclass
class MockBatchExecConfig:
    use_amp: bool = False
    logit_adjust_alpha: float = 1.0


class MockEngineRuntime:
    '''Mock engine runtime container.'''
    def __init__(self, model: MockMultiheadModel):
        state = state_mod.initialize_state(
            all_heads=['head_1'],
            batch_size=2,
            use_amp=False,
            device='cpu'
        )
        context = executor_mod.BatchExecContext(
            parent_map={},
            patch_per_blk=4,
            patch_per_dim=2,
            block_columns=2,
            device='cpu'
        )
        self.engine = executor_mod.BatchEngine(
            model=model,
            engine_state=state,
            config=MockBatchExecConfig(),
            context=context
        )
        self.engine_tasks = MockEngineTasks()

        # build optimization wrapper around model parameter
        opt_config = session_schema_optim_config()
        self.engine_optim = optim_builder.build_optimization(
            model,
            opt_config
        )


def session_schema_optim_config():
    '''Helper building standard dummy optimization config.'''
    @dataclasses.dataclass
    class _DummyOptimConfig:
        opt_cls: str = 'AdamW'
        lr: float = 1e-3
        weight_decay: float = 1e-4
        sched_cls: str | None = None
        sched_args: dict | None = None
        grad_clip_norm: float | None = 1.0

    return _DummyOptimConfig()


# ----- pytest fixtures
@pytest.fixture
def mock_model():
    return MockMultiheadModel()


@pytest.fixture
def mock_dataloaders():
    return MockDataLoaders()


@pytest.fixture
def mock_dispatcher():
    return MockDispatcher()


@pytest.fixture
def mock_runtime(mock_model):
    return MockEngineRuntime(mock_model)
