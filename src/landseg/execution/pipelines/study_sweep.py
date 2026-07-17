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

# standard imports
import datetime
import typing
# local imports
import landseg._constants as c
import landseg.artifacts as artifacts
import landseg.configs as configs
import landseg.core as core
import landseg.geopipe as geopipe
import landseg.models as models
import landseg.session as session
import landseg.study as study

# aliases
StepGenerator = typing.Generator[core.SessionStepSummary, None, None]
StepRunner: typing.TypeAlias = typing.Callable[..., StepGenerator]


def sweep(config: configs.RootConfig):
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
    # init run io folder tree
    paths = artifacts.ResultsPaths(f'{config.execution.exp_root}/results')
    paths.init(trace_to_last=False)

    # save running config per session
    config_ctrl = artifacts.Controller[dict](paths.config) # no policy
    config_ctrl.persist(config.as_dict)

    # init a SessionLogger
    logger = session.SessionLogger(
        name='session',
        log_file=paths.summary,
        console_lvl=None,
        enable_file_log=False
    )

    def run_wrapper():
        logger.init_summary(
            run_id=paths.run_id,
            pipeline=config.pipeline.name,
            start_time=datetime.datetime.now().strftime(c.TF_ISO8601)
        )
        logger.set_inputs(config.as_dict)
        try:
            logger.log_sep()

            # collect artifacts and build `DataSpecs`
            artifact_paths = artifacts.ArtifactPaths(
                f'{config.execution.exp_root}/artifacts/'
                f'{config.foundation.datablocks.name}'
            )
            dataspecs = geopipe.build_dataspec(
                artifact_paths,
                mode='default',
                ids_domain_name=config.dataspecs.domain_ids_name,
                vec_domain_name=config.dataspecs.domain_vec_name,
                print_out=False
            )

            # setup the model
            model = models.build_multihead_unet(
                patch_size=config.session.data_loader.patch_size,
                dataspecs=dataspecs,
                unet_backbone_config=config.models.unet_backbone_config,
                conditioning_config=config.models.conditioning_config,
                enable_clamp=config.models.numeric_safety.enable_clamp,
                clamp_range=config.models.numeric_safety.clamp_range
            )

            # build session runner
            session_context = session.SessionBuildContext(
                device=c.DEVICE,
                verbose_runner=False,
                session_paths=paths,
            )
            runner = session.factory.build_continous_training_session(
                dataspecs=dataspecs,
                model=model,
                config=config.session,
                context=session_context,
                logger=logger
            )

            yield from runner.run()

            # update summary
            logger.summary['completed_at'] = datetime.datetime.now().strftime(c.TF_ISO8601)
            logger.set_summary_status('SUCCESS')

        except Exception as e:
            logger.set_summary_status('FAILED')
            logger.log('ERROR', f'Trial execution failed: {e}', exc_info=True)
            raise e

        finally:
            logger.log_sep()
            logger.close() # summary dict will be persisted

    return paths.step_results, run_wrapper
