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

def make_objective(
    base_runner: typing.Callable[[typing.Any], float],
    cfg, # need a base type here
):
    '''doc'''

    def objective(trial: optuna.Trial) -> float:

        trial_cfg = _from_base_objectives(cfg, trial)

        return base_runner(trial_cfg)

    return objective

def _from_base_objectives(cfg, trial: optuna.Trial):
    '''doc'''

    _cfg = copy.deepcopy(cfg)
    # learning rate
    _cfg.session.components.optimization.lr = trial.suggest_float(
        name='lr',
        low=cfg.study.base.learning_rate[0],
        high=cfg.study.base.learning_rate[1],
        log=True
    )
    # weight decay
    _cfg.session.components.optimization.weight_decay = trial.suggest_float(
        name='weight_decay',
        low=cfg.study.base.weight_decay[0],
        high=cfg.study.base.weight_decay[1],
        log=True
    )
    # patch size
    _cfg.session.components.loader.patch_size = trial.suggest_int(
        name='patch_size',
        low=cfg.study.base.patch_size[0],
        high=cfg.study.base.patch_size[1],
        step=cfg.study.base.patch_size[2],
    )
    # batch size
    _cfg.session.components.loader.batch_size = trial.suggest_int(
        name='batch_size',
        low=cfg.study.base.batch_size[0],
        high=cfg.study.base.batch_size[1],
        step=cfg.study.base.batch_size[2],
    )

    return _cfg
