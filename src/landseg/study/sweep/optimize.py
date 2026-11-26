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
Optuna study execution utilities.

This module defines the thin integration layer between the project and
Optuna's study API. It is responsible for creating or resuming an Optuna
study with standard sampling and pruning policies, executing a bounded
number of trials, and returning the resulting Study object.
'''

# standard imports
import typing
# third-party imports
import optuna
# local imports
import landseg.configs as configs
import landseg.study.sweep as sweep

# -------------------------------Public Function-------------------------------
def run_sweep(
    runner_builder: typing.Callable,
    root_config: configs.RootConfig,
) -> optuna.Study:
    '''
    Create and run an Optuna study using the configured sweep policy.

    Constructs an Optuna `Study` using parameters defined in the sweep
    configuration, including storage, sampler, pruner, optimization
    direction, and trial count, then executes the optimization loop.

    The returned `Study` provides access to completed trials and
    metadata without performing post-hoc aggregation or ranking.

    Args:
        runner_builder:
            Callable accepting a `RootConfig` and returning a tuple of
            step-results output path and step generator runner.
        root_config:
            Root configuration providing sweep settings and base state.

    Returns:
        optuna.Study:
            Executed Optuna study instance containing trial history.
    '''
    # sweep config
    config = root_config.study.sweep

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
    objective = sweep.make_objective(runner_builder, root_config)

    # run optimization
    study.optimize(
        objective,
        n_trials=config.n_trials,
        show_progress_bar=True
    )

    # return the study instance
    return study
