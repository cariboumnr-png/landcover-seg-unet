# =========================================================================== #
#            Copyright © His Majesty the King in right of Ontario,            #
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

'''
This module mirrors the Hydra/YAML config tree using Python dataclasses,
suitable for OmegaConf structured configs.
'''

# standard imports
from __future__ import annotations
import dataclasses
import typing
# local imports
import landseg.configs.schema.base as base
from landseg.configs.schema.data import DataConfig
from landseg.configs.schema.models import ModelsConfig
from landseg.configs.schema.session import SessionConfig
from landseg.configs.schema.study import StudyConfig

# alias
field = dataclasses.field


@dataclasses.dataclass
class ExecutionContext(base.BaseConfigSection):
    '''Mutable execution context.'''
    verbosity: str | int | None = 'full' # 'full', 'logging_only', 'silent', 10/20/None
    exp_root: str = './experiment' # root directory for this experiment run
    user_cfg: str | None = None # external user configs
    dev_cfg: str | None = None # developer-only override config
    cli_mode: bool = False # indicates whether execution initiated via CLI
    preflight_target: str | None = None
    preflight_strict_mode: bool = False

    @property
    def console_level(self) -> int | None:
        '''Parse verbosity option into console level.'''
        if self.verbosity is None:
            return None
        if isinstance(self.verbosity, str):
            match self.verbosity:
                case 'full':
                    return 10
                case 'select':
                    return 20
                case 'silent':
                    return None
                case _:
                    raise ValueError(f'Invalid option: {self.verbosity}')
        match self.verbosity:
            case 10:
                return 10
            case 20:
                return 20
            case _:
                raise ValueError(f'Invalid option: {self.verbosity}')


@dataclasses.dataclass
class RootConfig(base.BaseConfigSection):
    '''Root structured config for landseg.'''
    # execution configs
    execution: ExecutionContext = field(default_factory=ExecutionContext)
    # data ETL settings
    data: DataConfig = field(default_factory=DataConfig)
    # model settings
    models: ModelsConfig = field(default_factory=ModelsConfig)
    # session settings
    session: SessionConfig = field(default_factory=SessionConfig)
    # study settings
    study: StudyConfig = field(default_factory=StudyConfig)
    # command
    command: str = 'default'

    @property
    def as_dict(self) -> dict[str, typing.Any]:
        return dataclasses.asdict(typing.cast(typing.Any, self))

    def validate(self) -> None:
        self.data.validate()
        self.models.validate()
        self.session.validate()
        self.study.validate()
