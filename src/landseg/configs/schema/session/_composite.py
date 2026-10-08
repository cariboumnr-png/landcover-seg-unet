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

# pylint: disable=missing-class-docstring

'''
Session config section
'''

# standard imports
import dataclasses
import typing
# local imports
import landseg.configs.schema.base as base
import landseg.configs.schema.session.dataloader as dataloader_sec
import landseg.configs.schema.session.engine as engine_sec
import landseg.configs.schema.session.orchestration as orchestration_sec

# ----- aliases
field = dataclasses.field
DataLoader = dataloader_sec.DataLoaderConfig
Engine = engine_sec.EngineConfig
Orchestration = orchestration_sec.OrchestrationConfig


@dataclasses.dataclass
class SessionConfig(base.BaseConfigSection):
    dataloader: DataLoader = field(default_factory=DataLoader)
    engine: Engine = field(default_factory=Engine)
    orchestration: Orchestration = field(default_factory=Orchestration)
    mode: str = 'continuous'
    output_dpath: str = '${execution.exp_root}/results/'

    def __post_init__(self):
        if self.mode == 'curriculum':
            self.orchestration.monitor.allow_early_stop = False
        # allow_early_stop=True is invalid for curriculum

    @property
    def engine_exec(self) -> engine_sec.EngineExec:
        '''Return config subgroup: `session.EngineExec`.'''
        return self.engine.engine_exec

    @property
    def engine_optim(self) -> engine_sec.EngineOptim:
        '''Return config subgroup: `session.EngineOptim`.'''
        return self.engine.engine_optim

    @property
    def engine_schedule(self) -> engine_sec.EngineSchedule:
        '''Return config subgroup: `session.EngineSchedule`.'''
        return self.engine.engine_schedule

    @property
    def engine_tasks(self) -> engine_sec.EngineTask:
        '''Return config subgroup: `session.EngineTask`.'''
        return self.engine.engine_tasks

    @property
    def training_mode(self) -> typing.Literal['continuous', 'curriculum']:
        '''Typed training mode.'''
        return typing.cast(typing.Literal['continuous', 'curriculum'], self.mode)

    def validate(self):
        self.dataloader.validate()
        self.engine.validate()
        self.orchestration.validate()

        if self.mode == 'continuous':
            if self.orchestration.curriculum.schema != 'single':
                raise ValueError(
                    '[curriculum.schema] must be "single" for a continuous '
                    'training session'
                )
        elif self.mode == 'curriculum':
            if self.orchestration.curriculum.schema == 'single':
                raise ValueError(
                    '[curriculum.schema] must not be "single" for a curriculum'
                    '-based training session; expected: "baseline" or "custom"'
                )
        else:
            raise ValueError(
                f'Invalid mode: {self.mode}, '
                f'must be "continuous" or "curriculum"'
            )
