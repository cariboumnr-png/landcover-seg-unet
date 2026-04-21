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

# standard imports
import typing
# third-party imports
import optuna
# local imports
import landseg.tuning as tuning

# -------------------------------Public Function-------------------------------
def run_study(
    runner: typing.Callable[[typing.Any], float],
    root_config: tuning.RootConfigShape,
) -> optuna.Study:
    '''doc'''

    # sweep config
    config = root_config.pipeline.study_sweep

    # storage (enables resume)
    storage = config.storage if config.storage is not None else None

    # sampler
    sampler = optuna.samplers.TPESampler(seed=config.seed)

    # pruner (default for TPE)
    pruner = optuna.pruners.MedianPruner(
        n_startup_trials=5,   # no pruning early trials
        n_warmup_steps=10,    # allow some learning signal
    )

    # create study
    study = optuna.create_study(
        storage=storage,
        sampler=sampler,
        pruner=pruner,
        study_name=config.study_name,
        direction=config.direction,
        load_if_exists=True,
    )

    # get objective
    objective = tuning.make_objective(runner, root_config)

    # run optimization
    study.optimize(
        objective,
        n_trials=config.n_trials,
        show_progress_bar=True
    )

    # return the study instance
    return study
