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

# third-party imports
import optuna
# local imports
import landseg.study.sweep as sweep
import landseg.study.sweep.presets as presets

def comp_quick(
    cfg: sweep.RootConfigShape,
    trial: optuna.Trial,
) -> sweep.RootConfigShape:
    '''Composite quick preset: optimizer + throughput.'''

    cfg = presets.obj_optimizer(cfg, trial)
    cfg = presets.obj_throughput(cfg, trial)
    return cfg

def comp_capacity(
    cfg: sweep.RootConfigShape,
    trial: optuna.Trial,
) -> sweep.RootConfigShape:
    '''Composite capacity preset: architecture + bottleneck.'''

    cfg = presets.obj_architecture(cfg, trial)
    cfg = presets.obj_bottleneck(cfg, trial)
    return cfg

def comp_mtl_quality(
    cfg: sweep.RootConfigShape,
    trial: optuna.Trial,
) -> sweep.RootConfigShape:
    '''Composite mtl quality preset: loss balance + regularization'''

    cfg = presets.obj_loss_balance(cfg, trial)
    cfg = presets.obj_loss_aux(cfg, trial)
    cfg = presets.obj_regularization(cfg, trial)
    return cfg

def comp_candidate(
    cfg: sweep.RootConfigShape,
    trial: optuna.Trial,
) -> sweep.RootConfigShape:
    '''Composite production candidate: combines all presets.'''

    cfg = presets.obj_optimizer(cfg, trial)
    cfg = presets.obj_throughput(cfg, trial)
    cfg = presets.obj_data_geometry(cfg, trial)
    cfg = presets.obj_context_window(cfg, trial)
    cfg = presets.obj_architecture(cfg, trial)
    cfg = presets.obj_bottleneck(cfg, trial)
    cfg = presets.obj_conditioning(cfg, trial)
    cfg = presets.obj_loss_balance(cfg, trial)
    cfg = presets.obj_loss_aux(cfg, trial)
    cfg = presets.obj_regularization(cfg, trial)
    return cfg
