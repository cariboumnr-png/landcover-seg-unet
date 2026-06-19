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

'''Loss and regularization preset objectives.'''

# standard imports
import copy
# third-party imports
import optuna
# local imports
import landseg.study.sweep as sweep

def obj_loss_balance(
    cfg: sweep.RootConfigShape,
    trial: optuna.Trial,
) -> sweep.RootConfigShape:
    '''
    Main loss balance mutations:
      - Focal loss weight (`float`)
      - Dice loss weight (`float`)
    '''

    trial_cfg = copy.deepcopy(cfg)
    study_cfg = cfg.study.loss_balance

    trial_cfg.set_objective_focal_weight(
        weight=trial.suggest_float(
            name='objective.focal_weight',
            low=study_cfg.focal_weight[0],
            high=study_cfg.focal_weight[1],
        )
    )

    trial_cfg.set_objective_dice_weight(
        weight=trial.suggest_float(
            name='objective.dice_weight',
            low=study_cfg.dice_weight[0],
            high=study_cfg.dice_weight[1],
        )
    )

    return trial_cfg

def obj_loss_aux(
    cfg: sweep.RootConfigShape,
    trial: optuna.Trial,
) -> sweep.RootConfigShape:
    '''
    Auxiliary loss mutations:
      - Spectral loss weight (`float`)
      - TV loss weight (`float`)
    '''

    trial_cfg = copy.deepcopy(cfg)
    study_cfg = cfg.study.regularization

    trial_cfg.set_objective_spectral_weight(
        weight=trial.suggest_float(
            name='objective.spectral_weight',
            low=study_cfg.spectral_weight[0],
            high=study_cfg.spectral_weight[1],
        )
    )

    trial_cfg.set_objective_tv_weight(
        weight=trial.suggest_float(
            name='objective.tv_weight',
            low=study_cfg.tv_weight[0],
            high=study_cfg.tv_weight[1],
        )
    )

    return trial_cfg

def obj_regularization(
    cfg: sweep.RootConfigShape,
    trial: optuna.Trial,
) -> sweep.RootConfigShape:
    '''
    Regularization mutations:
      - consistency regularizer lambda weight (`float`)
    '''

    trial_cfg = copy.deepcopy(cfg)
    study_cfg = cfg.study.mtl_consistency

    trial_cfg.set_mtl_consistency_lambda(
        value=trial.suggest_float(
            name='mtl.consistency_lambda',
            low=study_cfg.consistency_lambda[0],
            high=study_cfg.consistency_lambda[1],
        )
    )

    return trial_cfg
