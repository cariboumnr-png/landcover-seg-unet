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

# pylint: disable=too-many-branches

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
    recipe_dir: str = './configs' # external recipe directory path override
    knowledge_dir: str = './knowledge' # external knowledge directory override
    dev_cfg: str | None = None # developer-only override config
    preflight_target: str | None = None
    preflight_strict_mode: bool = False

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

    def validate(self, command: str | list[str] | None = None) -> None:
        '''Validate the entire configuration structure.'''
        to_validate = self.command if command is None else command
        need_to_validate = self.commands_to_validate(
            to_validate,
            self.execution.preflight_target
        )

        for cmd in need_to_validate:
            try:
                self.require_allowed_str(cmd, 'command', c.COMMANDS)
            except Exception as e:
                raise base.ConfigValidationError from e

            match cmd:
                case 'world-grid':
                    self.data.world_grid.validate()
                case 'data-harmonize':
                    self.data.harmonization.validate()
                case 'data-ingest':
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
                case 'e2e-intake':
                    self.data.harmonization.validate()
                    self.data.ingestion.validate()
                case 'e2e-experiment':
                    self.data.validate()
                    self.models.validate()
                    self.session.validate(mode='train')

    @staticmethod
    def commands_to_validate(
        command: str | list[str] | None,
        preflight_target: str | None,
    ) -> list[str]:
        '''Return a list of commands to be validated.'''
        if not command or command == 'default':
            return []

        if isinstance(command, list):
            return command

        if command == 'preflight':
            target = preflight_target
            if target is None or target == 'all':
                targets = list(c.SECTION_RECIPES) # validate all sections
            else:
                if target not in c.SECTION_RECIPES:
                    raise KeyError(
                        f'Target "{target}" not supported for preflight; '
                        f'allowed: {c.SECTION_RECIPES}'
                    )
                targets = [target] # only validate select section
            return targets

        return [command]
