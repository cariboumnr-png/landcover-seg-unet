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
Pipeline command schema specifications.

Defines sub-command configurations for training, evaluation, sweeps,
and preflight readiness checks.

Public APIs:
    - `CommandConfig`: Root command configuration section.
'''

# standard imports
import dataclasses


# ----- typing aliases
field = dataclasses.field


# ----- private dataclasses
@dataclasses.dataclass
class _TrainModel:

    def validate(self) -> None:...


@dataclasses.dataclass
class _PreflightConfig:
    target: str = 'all'
    strict: bool = False
    export_report: bool = True
    report_path: str | None = None
    check_gpu: bool = True

    def validate(self):...


# ----- public dataclasses
@dataclasses.dataclass
class CommandConfig:
    name: str = 'default'
    preflight: _PreflightConfig = field(default_factory=_PreflightConfig)
    model_train: _TrainModel = field(default_factory=_TrainModel)

    def validate(self):
        self.preflight.validate()
        self.model_train.validate()
