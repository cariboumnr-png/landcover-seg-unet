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
Study sweep execution entrypoints.

This module provides CLI-facing helpers for running Optuna-based sweep
studies and executing individual trials. It bridges project pipelines
with Optuna while preserving the invariant that each trial evaluates to
a single scalar objective.

Sweep orchestration is delegated to Optuna; study-level aggregation and
analysis are handled elsewhere.
'''

# local imports
import landseg.cli.pipelines as pipelines
import landseg.configs as configs
import landseg.study as study


def sweep(config: configs.RootConfig):
    '''
    Execute a configured study sweep.

    This function runs an Optuna study using the provided configuration
    and returns a small summary of the best observed result for CLI
    consumption. Full study inspection is performed separately by the
    study analysis pipeline.
    '''

    # run sweep and return
    s = study.run_sweep(_trial, config)
    return {
        'best_value': s.best_value,
        'best_params': s.best_params,
    }


def _trial(config: configs.RootConfig) -> float:
    '''
    Run a single sweep trial.

    This function executes one end-to-end training run using the provided
    configuration and returns a scalar objective value suitable for
    optimization. All detailed outputs are persisted as normal run
    artifacts and are not returned directly.

    Args:
        config: RootConfig with model, trainer, and runner settings.
    '''

    # set training runner to silent
    config.execution.verbosity = 'silent'

    # train
    meta = pipelines.train(config)

    # return the best value
    best = meta['summary'].get('best_value') # tighten later
    assert isinstance(best, float)
    return best
