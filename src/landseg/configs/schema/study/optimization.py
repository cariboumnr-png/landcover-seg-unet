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

'''Optimization hyperparameter search spaces for study sweeps.'''

# standard imports
import dataclasses
# local imports
import landseg.configs.schema.base as base

# alias
field = dataclasses.field


@dataclasses.dataclass
class Base(base.BaseConfigSection):
    '''Base optimization search space.'''
    learning_rate: tuple[float, float] = (1e-5, 1e-1)
    batch_size: tuple[int, int, int] = (16, 64, 16)


@dataclasses.dataclass
class Optimizer(base.BaseConfigSection):
    '''Optimizer search space'''
    learning_rate: tuple[float, float] = (1e-5, 1e-1)
    weight_decay: tuple[float, float] = (1e-6, 1e-2)


@dataclasses.dataclass
class Throughput(base.BaseConfigSection):
    '''Througput search space'''
    batch_size: tuple[int, int, int] = (16, 64, 16)
    use_amp: list[bool] = field(default_factory=lambda: [True, False])


@dataclasses.dataclass
class OptimizationSearchConfig(base.BaseConfigSection):
    '''Optimization hyperparameter search space configuration.'''
    base: Base = field(default_factory=Base)
    optimizer: Optimizer = field(default_factory=Optimizer)
    throughput: Throughput = field(default_factory=Throughput)

    def validate(self) -> None:
        self.base.validate()
        self.optimizer.validate()
        self.throughput.validate()
