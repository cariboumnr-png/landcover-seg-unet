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
# pylint: disable=missing-function-docstring

'''
Session schema
'''

# standard imports
import dataclasses
# local imports
import landseg.configs.schema.base as base

# alias
field = dataclasses.field


# ----- orchestration
@dataclasses.dataclass
class MonitorConfig(base.BaseConfigSection):
    metric_name: str = 'iou'
    track_heads: dict[str, float] | None = None
    track_mode: str = 'max'
    allow_early_stop: bool = True
    patience: int | None = 10
    min_delta: float | None = 0.0005

    def validate(self):
        self.require_attr_type_range('patience', int, (0, None))
        self.require_attr_type_range('min_delta', float, (0.0, None))


@dataclasses.dataclass
class PhaseConfig(base.BaseConfigSection):
    name: str = 'phase_0'
    num_epochs: int = 0
    start_epoch: int = 1 # 1-based
    lr_scale: float | None = 1.0
    active_heads: list[str] | None = None
    frozen_heads: list[str] | None = None

    def validate(self):
        self.require_attr_type_range('num_epochs', int, (1, None))
        self.require_attr_type_range('start_epochs', int, (1, self.num_epochs))
        self.require_attr_type_range('lr_scale', float, (0.0, None))


@dataclasses.dataclass
class SinglePhase(base.BaseConfigSection):
    name: str = 'single'
    phases: list[PhaseConfig] = field(default_factory=lambda: [PhaseConfig()])


@dataclasses.dataclass
class BaselinePhases(base.BaseConfigSection):
    name: str = 'baseline'
    phases: list[PhaseConfig] = field(default_factory=lambda: [PhaseConfig()])


@dataclasses.dataclass
class CustomPhases(base.BaseConfigSection):
    name: str = 'custom'
    phases: list[PhaseConfig] = field(default_factory=lambda: [PhaseConfig()])


@dataclasses.dataclass
class Curriculum(base.BaseConfigSection):
    schema: str = 'single'
    single: SinglePhase = field(default_factory=SinglePhase)
    baseline: BaselinePhases = field(default_factory=BaselinePhases)
    custom: CustomPhases = field(default_factory=CustomPhases)


@dataclasses.dataclass
class OrchestrationConfig(base.BaseConfigSection):
    monitor: MonitorConfig = field(default_factory=MonitorConfig)
    curriculum: Curriculum = field(default_factory=Curriculum)
    resume_from_last: bool = False

    @property
    def single_phase(self) -> PhaseConfig:
        '''Single phase from runtime configs.'''
        return self.curriculum.single.phases[0]

    @property
    def multi_phases(self) -> list[PhaseConfig]:
        '''List of phases by configs.'''
        # currently supported pre-configured phases
        schema = self.curriculum.schema
        match schema:
            case 'baseline':
                return self.curriculum.baseline.phases
            case 'custom':
                return self.curriculum.custom.phases
            case _:
                raise base.ConfigValidationError(
                    f'Invalid multi-phases schema: {schema}'
                )

    def validate(self):
        self.monitor.validate()
        if self.curriculum.schema == 'single':
            self.single_phase.validate()
        else:
            for phase in self.multi_phases:
                phase.validate()
