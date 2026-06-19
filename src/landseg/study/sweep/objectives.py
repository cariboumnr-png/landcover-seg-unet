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
Optuna objective construction utilities.

This module defines the minimal contract between Optuna-based sweep
orchestration and the project runtime. It translates a static root
configuration into per-trial variants by applying parameter suggestions
from an Optuna Trial, while preserving the invariant that each trial
evaluates to a single scalar objective.

Richer cross-run analysis, multi-metric reporting, and study-level
aggregation are intentionally out of scope and belong to the study
analysis layer defined in ADR-0026.
'''

# landseg/tuning/objective.py

# standard imports
import typing
# third-party imports
import optuna
# local imports
import landseg.artifacts as artifacts
import landseg.core as core
import landseg.study.sweep as sweep
import landseg.study.sweep.presets as presets

# aliases
StepGenerator = typing.Generator[core.SessionStepSummary, None, None]
StepRunner: typing.TypeAlias = typing.Callable[..., StepGenerator]

# -------------------------------Public Function-------------------------------
def make_objective(
    runner_builder: typing.Callable[..., tuple[str, StepRunner]],
    cfg: sweep.RootConfigShape,
) -> typing.Callable[[optuna.Trial], float]:
    '''
    Build an Optuna-compatible objective function from a base runner.

    This function binds a project-specific execution callable
    (`base_runner`) to an Optuna Trial by generating a trial-specific
    configuration according to the configured study objective. The
    resulting callable conforms to Optuna's `(Trial) -> float`
    interface and returns a single scalar value used for optimization.

    Parameter selection logic is intentionally minimal and delegated to
    objective-specific helpers to avoid coupling sweep orchestration
    with runtime or analysis concerns.
    '''

    # Optuna objective invoked once per trial
    def objective(trial: optuna.Trial) -> float:

        # get trial config depending on the objective preset
        objectives_fn = presets.resolve(cfg.pipeline.study_sweep.preset_name)
        trial_cfg = objectives_fn(cfg, trial)

        # build the runner with trial config
        step_results_path, run = runner_builder(trial_cfg)

        # last metric tracking
        last_value = 0.0

        # step results tracking and persisting
        ctrl = artifacts.Controller[list[dict]](step_results_path)
        steps: list[dict] = []

        # drive the runner
        for step in run():

            # get value from step
            value = step.val_metrics_value
            last_value = value

            # report intermediate result
            trial.report(value, step.epoch_in_phase)

            # collect step results
            steps.append(step.as_dict)

            # check pruning condition
            if trial.should_prune():
                raise optuna.TrialPruned()

        # persist the step results JSON
        ctrl.persist(steps)

        # return the last value
        return last_value

    return objective
