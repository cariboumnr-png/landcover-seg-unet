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

'''Fixtures for testing `landseg.session` module.'''

# standard imports
import dataclasses
# third-party imports
import pytest
import torch
# local imports
import landseg.configs.schema.sections.session as session_schema


# ----- `_DummyModel` helper class
class _DummyModel(torch.nn.Module):

    def __init__(self, spatial_divisor: int = 16):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 2, kernel_size=1)
        self.linear = torch.nn.Linear(2, 2)
        self.spatial_divisor = spatial_divisor
        self.logit_adjust_alpha = 1.0

    def set_active_heads(self, active_heads: list[str] | None) -> None:
        _ = active_heads

    def set_frozen_heads(self, frozen_heads: list[str] | None) -> None:
        _ = frozen_heads

    def reset_heads(self) -> None:
        pass

    def set_logit_adjust_alpha(self, alpha: float) -> None:
        self.logit_adjust_alpha = alpha

    def forward(self, x, ids_domain=None, vec_domain=None):
        _ = ids_domain, vec_domain
        if x.dim() == 2:
            return self.linear(x)
        return {'head_1': self.conv(x)}


# ----- mock model and optimizer fixtures
@pytest.fixture
def mock_model():
    return _DummyModel()


# ----- session config and constraint fixtures
@pytest.fixture
def session_config():
    return session_schema.SessionConfig()


# ----- dataloader and logger helper classes
@dataclasses.dataclass
class _MockDataloaderMeta:
    batch_size: int = 2
    patch_size: int = 16
    preview_context: None = None


@dataclasses.dataclass
class _MockDataLoaders:
    meta: _MockDataloaderMeta = dataclasses.field(
        default_factory=_MockDataloaderMeta
    )
    train: list = dataclasses.field(default_factory=list)
    val: list = dataclasses.field(default_factory=list)
    test: list = dataclasses.field(default_factory=list)


class _MockLogger:
    console_lvl = 'INFO'


@dataclasses.dataclass
class _MockSessionPaths:
    logs: str = 'dummy_logs_dir'


# ----- dataloader, logger, and paths fixtures
@pytest.fixture
def mock_dataloaders():
    return _MockDataLoaders()


@pytest.fixture
def mock_logger():
    return _MockLogger()


@pytest.fixture
def mock_session_paths(tmp_path):
    return _MockSessionPaths(logs=str(tmp_path / 'logs'))
