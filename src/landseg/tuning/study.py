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
Optuna study.
'''

# third-party imports
import optuna
# local imports
import landseg.tuning as tuning


def run_study(
    runner,
    cfg
):
    '''doc'''
    sampler = optuna.samplers.TPESampler(seed=cfg.pipeline.study_sweep.seed)

    study = optuna.create_study(
        direction=cfg.pipeline.study_sweep.direction,
        sampler=sampler,
    )

    objective = tuning.make_objective(runner, cfg)

    study.optimize(
        objective,
        n_trials=cfg.pipeline.study_sweep.n_trials,
    )

    return study
