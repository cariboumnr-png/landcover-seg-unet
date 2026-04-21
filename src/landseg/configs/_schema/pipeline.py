# =========================================================================== #
#           Copyright (c) His Majesty the King in right of Ontario,           #
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
Pipieline schema
'''

# standard imports
import dataclasses

# alias
field = dataclasses.field

# ------------------------------PIPELINE  CONFIGS------------------------------
@dataclasses.dataclass
class _TrainModelConfig:
    pass  # training uses session config only (for now)

@dataclasses.dataclass
class _EvaluateModelConfig:
    checkpoint: str | None = None
    split: str = 'test'
    export_previews: bool = False

@dataclasses.dataclass
class PipelineConfig:
    name: str = 'default'
    model_train: _TrainModelConfig = field(default_factory=_TrainModelConfig)
    model_evaluate: _EvaluateModelConfig = field(default_factory=_EvaluateModelConfig)
