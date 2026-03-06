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

'''
Collection of protocol-typed components used by the training loop.

This small module exists to centralize the set of objects the trainer
requires (model, dataloaders, losses, metrics, optimization, callbacks),
while keeping their concrete implementations defined in other modules.
'''

# standard imports
import dataclasses
# local imports
import landseg.training.common as common

# ------------------------------Public  Dataclass------------------------------
@dataclasses.dataclass
class TrainerComponents:
    '''
    Container for all trainer-required components.

    Each field uses a protocol type from `landseg.training.common`,
    allowing different concrete implementations to be supplied while
    keeping the trainer strongly typed and modular.
    '''
    model: common.MultiheadModelLike
    dataloaders: common.DataLoadersLike
    headspecs: common.HeadSpecsLike
    headlosses: common.HeadLossesLike
    headmetrics: common.HeadMetricsLike
    optimization: common.OptimizationLike
    callbacks: common.CallBacksLike
