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
import dataclasses
# local imports
import landseg._constants as c
import landseg.configs.schema.base as base
import landseg.configs.schema.data as data
import landseg.configs.schema.models as models
import landseg.configs.schema.session as session
import landseg.configs.schema.study as study

# alias
field = dataclasses.field
DataConfig = data.DataConfig
ModelsConfig = models.ModelsConfig
SessionConfig = session.SessionConfig
StudyConfig = study.StudyConfig


@dataclasses.dataclass
class ExecutionContext(base.BaseConfigSection):
    '''Mutable execution context.'''
    verbosity: str | int | None = 'full'
    exp_root: str = './experiment'
    preflight_target: str | None = None
    preflight_strict_mode: bool = False
    dev_cfg: str | None = None # developer-only override config
    recipe_fpath: str | None = None # external recipe file path override
    cli_mode: bool = False # indicates whether execution initiated via CLI

    @property
    def console_level(self) -> int | None:
        '''Normalized console level from verbosity setting.'''
        match self.verbosity:
            case 10: return 10
            case 20: return 20
            case 'full': return 10
            case 'select': return 20
            case 'silent': return None
            case None: return None
            case _: raise ValueError(f'Invalid option: {self.verbosity}')


@dataclasses.dataclass
class RootConfig(base.BaseConfigSection):
    '''Root structured configuration schema.'''
    execution: ExecutionContext = field(default_factory=ExecutionContext)
    data: DataConfig = field(default_factory=DataConfig)
    models: ModelsConfig = field(default_factory=ModelsConfig)
    session: SessionConfig = field(default_factory=SessionConfig)
    study: StudyConfig = field(default_factory=StudyConfig)
    command: str = 'default'

    def validate(self, command: str | None = None) -> None:
        '''Validate the entire configuration structure.'''
        command = self.command if command is None else command
        try:
            self.require_type_range(command, 'command', str, c.COMMANDS)
        except ValueError as e:
            raise base.ConfigValidationError(
                f'Failed to validate command: {command}'
            ) from e
        match command:
            case 'default' | 'preflight':
                pass
            case 'world-grid':
                self.data.world_grid.validate()
            case 'data-harmonize':
                self.data.harmonization.validate()
            case 'data-ingest' | 'batch-ingest':
                self.data.ingestion.validate()
            case 'data-prepare':
                self.data.preparation.validate()
            case 'model-train':
                self.models.validate()
                self.session.validate(mode='train')
            case 'diagnose-overfit':
                self.models.validate()
                self.session.dataloader.validate()
                self.session.engine.validate()
            case 'model-evaluate':
                self.models.validate()
                self.session.validate(mode='evaluate')
            case 'study-sweep':
                self.models.validate()
                self.session.validate(mode='train')
                self.study.validate()
            case 'study-analysis':
                self.study.sweep.validate()
            case 'e2e-intake' | 'end-to-end-intake':
                self.data.harmonization.validate()
                self.data.ingestion.validate()
            case 'e2e-experiment' | 'end-to-end-experiment':
                self.data.validate()
                self.models.validate()
                self.session.validate(mode='train')
