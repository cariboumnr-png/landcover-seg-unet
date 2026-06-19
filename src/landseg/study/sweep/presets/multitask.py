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

'''Multi-task preset objectives.'''

# standard imports
import copy
# third-party imports
import optuna
# local imports
import landseg.study.sweep as sweep

def obj_head_weights(
    cfg: sweep.RootConfigShape,
    trial: optuna.Trial,
) -> sweep.RootConfigShape:
    '''
    Head weights preset mutations:
      - logit adjust alpha weight (`float`)
    '''

    trial_cfg = copy.deepcopy(cfg)
    study_cfg = cfg.study.head_weights

    trial_cfg.set_runtime_logit_adjust_alpha(
        alpha=trial.suggest_float(
            name='runtime.logit_adjust_alpha',
            low=study_cfg.logit_adjust_alpha[0],
            high=study_cfg.logit_adjust_alpha[1],
        )
    )

    return trial_cfg

def obj_mtl_joint(
    cfg: sweep.RootConfigShape,
    trial: optuna.Trial,
) -> sweep.RootConfigShape:
    '''
    MTL joint preset mutations:
      - consistency lambda (`float`)
      - logit adjust alpha (`float`)
    '''

    trial_cfg = copy.deepcopy(cfg)
    study_cfg = cfg.study.mtl_joint

    trial_cfg.set_mtl_consistency_lambda(
        value=trial.suggest_float(
            name='mtl.consistency_lambda',
            low=study_cfg.consistency_lambda[0],
            high=study_cfg.consistency_lambda[1],
        )
    )

    trial_cfg.set_runtime_logit_adjust_alpha(
        alpha=trial.suggest_float(
            name='runtime.logit_adjust_alpha',
            low=study_cfg.logit_adjust_alpha[0],
            high=study_cfg.logit_adjust_alpha[1],
        )
    )

    return trial_cfg

def obj_hierarchy(
    cfg: sweep.RootConfigShape,
    trial: optuna.Trial,
) -> sweep.RootConfigShape:
    '''
    Hierarchy preset mutations:
      - consistency lambda (`float`)
      - consistency reduction (`str`)
    '''

    trial_cfg = copy.deepcopy(cfg)
    study_cfg = cfg.study.hierarchy

    trial_cfg.set_mtl_consistency_lambda(
        value=trial.suggest_float(
            name='mtl.consistency_lambda',
            low=study_cfg.consistency_lambda[0],
            high=study_cfg.consistency_lambda[1],
        )
    )

    trial_cfg.set_mtl_consistency_reduction(
        reduction=trial.suggest_categorical(
            name='mtl.consistency_reduction',
            choices=study_cfg.consistency_reduction,
        )
    )

    return trial_cfg
