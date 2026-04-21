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
Optuna objectives
'''

# landseg/tuning/objective.py

# standard imports
import copy
import typing
# third-party imports
import optuna
# local imports
import landseg.tuning as tuning

# -------------------------------Public Function-------------------------------
def make_objective(
    base_runner: typing.Callable[[typing.Any], float],
    cfg: tuning.RootConfigShape,
) -> typing.Callable[[optuna.Trial], float]:
    '''doc'''

    # optuna objective function
    def objective(trial: optuna.Trial) -> float:
        # get trial config depending on objectives
        obj = cfg.pipeline.study_sweep.objective
        match obj:
            case 'base': trial_cfg = _from_base_objectives(cfg, trial)
            case _: raise ValueError(f'Invalid objective: {obj}')
        # return objective
        return base_runner(trial_cfg)

    return objective

# ------------------------------private  function------------------------------
def _from_base_objectives(
    cfg: tuning.RootConfigShape,
    trial: optuna.Trial
) -> tuning.RootConfigShape:
    '''doc'''

    trial_cfg = copy.deepcopy(cfg)
    # learning rate
    trial_cfg.set_lr(
        lr=trial.suggest_float(
            name='lr',
            low=cfg.study.base.learning_rate[0],
            high=cfg.study.base.learning_rate[1],
            log=True
        )
    )
    # weight decay
    trial_cfg.set_weight_decay(
        weight_decay=trial.suggest_float(
            name='weight_decay',
            low=cfg.study.base.weight_decay[0],
            high=cfg.study.base.weight_decay[1],
            log=True
        )
    )
    # patch size
    trial_cfg.set_patch_size(
        patch_size=trial.suggest_int(
            name='patch_size',
            low=cfg.study.base.patch_size[0],
            high=cfg.study.base.patch_size[1],
            step=cfg.study.base.patch_size[2],
        )
    )
    # batch size
    trial_cfg.set_batch_size(
        batch_size=trial.suggest_int(
            name='batch_size',
            low=cfg.study.base.batch_size[0],
            high=cfg.study.base.batch_size[1],
            step=cfg.study.base.batch_size[2],
        )
    )

    return trial_cfg
