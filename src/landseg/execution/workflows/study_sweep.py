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
Study sweep execution entrypoints.

This module provides CLI-facing helpers for running Optuna-based sweep
studies and executing individual trials. It bridges project pipelines
with Optuna while preserving the invariant that each trial evaluates to
a single scalar objective.

Sweep orchestration is delegated to Optuna; study-level aggregation and
analysis are handled elsewhere.
'''

# standard imports
import typing
# local imports
import landseg.artifacts as artifacts
import landseg.configs as configs
import landseg.core as core
import landseg.execution.pipelines as pipelines
import landseg.study as study

# aliases
StepGenerator = typing.Generator[core.SessionStepSummary, None, None]
StepRunner: typing.TypeAlias = typing.Callable[..., StepGenerator]


def execute_study_sweep(config: configs.RootConfig):
    '''
    Execute a configured study sweep.

    This function runs an Optuna study using the provided configuration
    and returns a small summary of the best observed result for CLI
    consumption. Full study inspection is performed separately by the
    study analysis pipeline.
    '''
    # run sweep and return
    s = study.run_sweep(_runner_builder, config)
    return {
        'best_value': s.best_value,
        'best_params': s.best_params,
    }


def _runner_builder(config: configs.RootConfig) -> tuple[str, StepRunner]:
    '''Build a continuous training session runner.'''
    config.session.mode = 'continuous' # ensure session type

    artifact_paths = artifacts.ArtifactPaths.from_config(config)

    def run_wrapper():
        training_pipeline = pipelines.ModelTraining(
            config,
            artifact_paths=artifact_paths,
            disable_console_logging=True
        )

        runner = training_pipeline.build_session_runner(mode_override='continuous')
        logger = training_pipeline.logger
        assert logger is not None

        try:
            yield from runner.run()

        except Exception as e:
            logger.set_summary_status('FAILED')
            logger.log('ERROR', f'Trial execution failed: {e}', exc_info=True)
            raise e

        finally:
            logger.set_summary_status('SUCCESS')
            logger.log_sep()
            logger.close() # summary dict will be persisted

    return artifact_paths.session.step_results, run_wrapper
