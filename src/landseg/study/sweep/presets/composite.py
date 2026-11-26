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
Composite preset objectives combining multiple parameter-group presets.

Public APIs:
    - `comp_candidate`: Compose all production candidate search presets.
    - `comp_capacity`: Compose architecture and bottleneck presets.
    - `comp_mtl_quality`: Compose loss and regularization presets.
    - `comp_quick`: Compose optimizer and throughput presets.
'''

# third-party imports
import optuna
# local imports
import landseg.study.sweep as sweep
import landseg.study.sweep.presets as presets

def comp_quick(
    cfg: sweep.TrialMutator,
    trial: optuna.Trial,
) -> sweep.TrialMutator:
    '''
    Apply quick sweep presets combining optimizer and throughput.

    Args:
        cfg:
            Mutable trial configuration wrapper.
        trial:
            Active Optuna trial providing parameter suggestions.

    Returns:
        sweep.TrialMutator:
            Updated trial mutator instance.
    '''
    cfg = presets.obj_optimizer(cfg, trial)
    cfg = presets.obj_throughput(cfg, trial)
    return cfg

def comp_capacity(
    cfg: sweep.TrialMutator,
    trial: optuna.Trial,
) -> sweep.TrialMutator:
    '''
    Apply capacity sweep presets combining architecture and bottleneck.

    Args:
        cfg:
            Mutable trial configuration wrapper.
        trial:
            Active Optuna trial providing parameter suggestions.

    Returns:
        sweep.TrialMutator:
            Updated trial mutator instance.
    '''
    cfg = presets.obj_architecture(cfg, trial)
    cfg = presets.obj_bottleneck(cfg, trial)
    return cfg

def comp_mtl_quality(
    cfg: sweep.TrialMutator,
    trial: optuna.Trial,
) -> sweep.TrialMutator:
    '''
    Apply MTL quality presets combining loss balance and regularization.

    Args:
        cfg:
            Mutable trial configuration wrapper.
        trial:
            Active Optuna trial providing parameter suggestions.

    Returns:
        sweep.TrialMutator:
            Updated trial mutator instance.
    '''
    cfg = presets.obj_loss_balance(cfg, trial)
    cfg = presets.obj_loss_aux(cfg, trial)
    cfg = presets.obj_regularization(cfg, trial)
    return cfg

def comp_candidate(
    cfg: sweep.TrialMutator,
    trial: optuna.Trial,
) -> sweep.TrialMutator:
    '''
    Apply full production candidate presets across all search domains.

    Args:
        cfg:
            Mutable trial configuration wrapper.
        trial:
            Active Optuna trial providing parameter suggestions.

    Returns:
        sweep.TrialMutator:
            Updated trial mutator instance.
    '''
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
