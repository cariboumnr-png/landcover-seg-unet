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
import copy
import typing
# third-party imports
import optuna
# local imports
import landseg.core as core
import landseg.study.sweep as sweep

# aliases
StepGenerator = typing.Generator[core.TrainingSessionStep, None, None]
StepRunner: typing.TypeAlias = typing.Callable[..., StepGenerator]

# -------------------------------Public Function-------------------------------
def make_objective(
    runner_builder: typing.Callable[..., StepRunner],
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
        # get trial config depending on objectives
        obj = cfg.pipeline.study_sweep.objective
        match obj:
            case 'base': trial_cfg = _from_base_objectives(cfg, trial)
            case _: raise ValueError(f'Invalid objective: {obj}')

        # build the runner with trial config
        run = runner_builder(trial_cfg)

        # last metric tracking
        last_value = 0.0

        # drive the runner
        for step in run():

            # get value from step
            value = step.objective_value
            last_value = value

            # report intermediate result
            trial.report(value, step.epoch)

            # check pruning condition
            if trial.should_prune():
                raise optuna.TrialPruned()

        # return the last value
        return last_value

    return objective

# ------------------------------private  function------------------------------
def _from_base_objectives(
    cfg: sweep.RootConfigShape,
    trial: optuna.Trial
) -> sweep.RootConfigShape:
    '''Derive a trial-specific config from base study objectives.'''

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
    # return trial config
    return trial_cfg
