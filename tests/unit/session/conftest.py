# =========================================================================== #
#           Copyright © His Majesty the King in right of Ontario,           #
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

'''Fixtures for testing `landseg.session` module.'''

# third-party imports
import pytest
import torch
# local imports
import landseg.artifacts as artifacts
import landseg.configs.schema.sections.session as session_schema
import landseg.session.data.loader as data_loader
import landseg.session.engine.batch as batch_mod
import landseg.session.engine.epoch as epoch_mod
import landseg.session.engine.epoch.policy.evaluator as eval_mod
import landseg.session.engine.epoch.policy.trainer as trainer_mod
import landseg.session.engine.optim as optim_mod
import landseg.session.engine.tasks as tasks_mod
import landseg.session.instrumentation.callbacks as callbacks_mod


@pytest.fixture
def mock_model():
    return _MockMultiheadModel()


@pytest.fixture
def session_config():
    return session_schema.SessionConfig()


@pytest.fixture
def mock_session_paths(tmp_path):
    return artifacts.SessionPaths(f'{tmp_path}/temp_results/')


@pytest.fixture
def mock_dataloaders(dataspecs, session_config):
    return data_loader.build_dataloaders(dataspecs, session_config.data_loader)


@pytest.fixture
def mock_runtime(dataspecs, mock_dataloaders, mock_model, session_config):
    batch_engine = batch_mod.build_batch_engine(
        dataspecs=dataspecs,
        dataloaders=mock_dataloaders,
        model=mock_model,
        config=session_config.engine_exec,
        device='cpu',
    )
    optimization = optim_mod.build_optimization(
        model=mock_model,
        config=session_config.engine_optim,
    )
    engine_tasks = tasks_mod.build_engine_tasks(
        dataspecs,
        session_config.engine_tasks,
    )
    return epoch_mod.EngineRuntime(
        engine=batch_engine,
        engine_optim=optimization,
        engine_tasks=engine_tasks,
    )


@pytest.fixture
def mock_dispatcher():
    return callbacks_mod.CallbackDispatcher([])


@pytest.fixture
def mock_evaluator(mock_runtime, mock_dataloaders, mock_dispatcher):
    return eval_mod.MultiHeadEvaluator(
        val_every=1,
        infer_every=1,
        dataset='val',
        engine_runtime=mock_runtime,
        dataloaders=mock_dataloaders,
        dispatcher=mock_dispatcher,
        device='cpu'
    )


@pytest.fixture
def mock_trainer(mock_runtime, mock_dataloaders, mock_dispatcher):
    return trainer_mod.MultiHeadTrainer(
        update_every=1,
        engine_runtime=mock_runtime,
        dataloaders=mock_dataloaders,
        dispatcher=mock_dispatcher,
        device='cpu'
    )

# ----- mock helper classes
class _MockMultiheadModel(torch.nn.Module):
    '''Mock multihead model for engine epoch tests.'''
    def __init__(self, spatial_divisor: int = 16):
        super().__init__()
        self.conv = torch.nn.Conv2d(4, 2, kernel_size=1)
        self.linear = torch.nn.Linear(2, 2)
        self.spatial_divisor = spatial_divisor
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
        if x.dim() == 2:
            return self.linear(x)
        assert self.active_heads is not None
        return {head: self.conv(x) for head in self.active_heads}
