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
    '''Simple module exposing parameters and head control protocol stubs for testing.'''

    def __init__(self, spatial_divisor: int = 16):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 2, kernel_size=1)
        self.linear = torch.nn.Linear(2, 2)
        self.spatial_divisor = spatial_divisor

    def set_active_heads(self, active_heads: list[str] | None) -> None:
        _ = active_heads

    def set_frozen_heads(self, frozen_heads: list[str] | None) -> None:
        _ = frozen_heads

    def reset_heads(self) -> None:
        pass

    def set_logit_adjust_alpha(self, alpha: float) -> None:
        _ = alpha

    def forward(self, x, ids_domain=None, vec_domain=None):
        _ = ids_domain, vec_domain
        if x.dim() == 2:
            return self.linear(x)
        return {'head_1': self.conv(x)}


# ----- mock model and optimizer fixtures
@pytest.fixture
def mock_model():
    '''Return a `_DummyModel` instance for testing.'''
    return _DummyModel()

# ----- session config and constraint fixtures
@pytest.fixture
def session_config():
    '''Return a default `SessionConfig` schema instance.'''
    return session_schema.SessionConfig()


@pytest.fixture
def mock_constraint():
    '''Return a factory creating dummy `_MTLConstraints` instances.'''
    def _create(
        name: str = 'rule_1',
        source_head: str = 'head_1',
        trigger_val: int = 1,
        target_head: str = 'head_2',
        forbidden: list[int] | None = None
    ):
        if forbidden is None:
            forbidden = [2]
        return session_schema._MTLConstraints(
            name=name,
            source_head=source_head,
            trigger_val=trigger_val,
            target_head=target_head,
            forbidden=forbidden
        )
    return _create


# ----- dataloader and logger helper classes
@dataclasses.dataclass
class _MockDataloaderMeta:
    '''Mock metadata container for dataloaders.'''
    batch_size: int = 2
    patch_size: int = 16
    preview_context: None = None


@dataclasses.dataclass
class _MockDataLoaders:
    '''Mock dataloaders container matching `DataLoadersLike`.'''
    meta: _MockDataloaderMeta = dataclasses.field(
        default_factory=_MockDataloaderMeta
    )
    train: list = dataclasses.field(default_factory=list)
    val: list = dataclasses.field(default_factory=list)
    test: list = dataclasses.field(default_factory=list)


class _MockLogger:
    '''Mock logger instance for session tests.'''
    console_lvl = 'INFO'


@dataclasses.dataclass
class _MockSessionPaths:
    '''Mock session results paths manager.'''
    logs: str = 'dummy_logs_dir'


# ----- dataloader, logger, and paths fixtures
@pytest.fixture
def mock_dataloaders():
    '''Return a `_MockDataLoaders` instance for testing.'''
    return _MockDataLoaders()


@pytest.fixture
def mock_logger():
    '''Return a `_MockLogger` instance for testing.'''
    return _MockLogger()


@pytest.fixture
def mock_session_paths(tmp_path):
    '''Return a `_MockSessionPaths` instance bound to a temporary directory.'''
    return _MockSessionPaths(logs=str(tmp_path / 'logs'))
