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
Composite preset objectives combining multiple parameter-group presets.
'''

from __future__ import annotations
import optuna
import landseg.study.sweep as sweep

def quick_objectives(
    cfg: sweep.RootConfigShape,
    trial: optuna.Trial,
) -> sweep.RootConfigShape:
    '''Composite quick preset: optimizer + throughput.'''

    from .optimizer import optimizer_objectives, throughput_objectives

    cfg = optimizer_objectives(cfg, trial)
    cfg = throughput_objectives(cfg, trial)
    return cfg

def capacity_objectives(
    cfg: sweep.RootConfigShape,
    trial: optuna.Trial,
) -> sweep.RootConfigShape:
    '''Composite capacity preset: architecture + bottleneck.'''

    from .architecture import architecture_objectives, bottleneck_objectives

    cfg = architecture_objectives(cfg, trial)
    cfg = bottleneck_objectives(cfg, trial)
    return cfg

def mtl_quality_objectives(
    cfg: sweep.RootConfigShape,
    trial: optuna.Trial,
) -> sweep.RootConfigShape:
    '''
    Composite mtl quality preset: loss balance + regularization +
    mtl consistency.
    '''

    from .losses import (
        loss_balance_objectives,
        regularization_objectives,
        mtl_consistency_objectives,
    )

    cfg = loss_balance_objectives(cfg, trial)
    cfg = regularization_objectives(cfg, trial)
    cfg = mtl_consistency_objectives(cfg, trial)
    return cfg

def production_candidate_objectives(
    cfg: sweep.RootConfigShape,
    trial: optuna.Trial,
) -> sweep.RootConfigShape:
    '''
    Composite production candidate: combines all presets.
    '''

    from .optimizer import optimizer_objectives, throughput_objectives
    from .data import data_geometry_objectives, context_window_objectives
    from .architecture import (
        architecture_objectives,
        bottleneck_objectives,
        conditioning_objectives,
    )
    from .losses import (
        loss_balance_objectives,
        regularization_objectives,
        mtl_consistency_objectives,
    )

    cfg = optimizer_objectives(cfg, trial)
    cfg = throughput_objectives(cfg, trial)
    cfg = data_geometry_objectives(cfg, trial)
    cfg = context_window_objectives(cfg, trial)
    cfg = architecture_objectives(cfg, trial)
    cfg = bottleneck_objectives(cfg, trial)
    cfg = conditioning_objectives(cfg, trial)
    cfg = loss_balance_objectives(cfg, trial)
    cfg = regularization_objectives(cfg, trial)
    cfg = mtl_consistency_objectives(cfg, trial)
    return cfg

