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
import typing
# local imports
import landseg._constants as c
import landseg.artifacts as artifacts
import landseg.configs as configs
import landseg.core as core
import landseg.geopipe as geopipe
import landseg.models as models
import landseg.session as session
import landseg.utils as utils
import landseg.study as study

# aliases
StepGenerator = typing.Generator[core.TrainingSessionStep, None, None]
StepRunner: typing.TypeAlias = typing.Callable[..., StepGenerator]

#
def sweep(config: configs.RootConfig):
    '''
    Execute a configured study sweep.

    This function runs an Optuna study using the provided configuration
    and returns a small summary of the best observed result for CLI
    consumption. Full study inspection is performed separately by the
    study analysis pipeline.
    '''

    # run sweep and return
    s = study.run_sweep(_build_runner, config)
    return {
        'best_value': s.best_value,
        'best_params': s.best_params,
    }

def _build_runner(config: configs.RootConfig) -> StepRunner:
    '''Build a continuous training orchestraction runner.'''

    # init run io folder tree
    paths = artifacts.ResultsPaths(f'{config.execution.exp_root}/results')
    paths.init(trace_to_last=config.session.resume_from_last)

    # create a centralized main logger
    logger = utils.Logger(
        name='main',
        log_file=paths.main_log_file,
        console_lvl=None
    )

    # collect artifacts and build dataspsec
    artifact_paths=artifacts.ArtifactPaths(f'{config.execution.exp_root}/artifacts')

    # collect artifacts and build dataspsec
    dataspecs = geopipe.build_dataspec(
        artifact_paths,
        mode='default',
        ids_domain_name=config.dataspecs.domain_ids_name,
        vec_domain_name=config.dataspecs.domain_vec_name,
        print_out=False
    )
    # setup the model
    model = models.build_multihead_unet(
        dataspecs=dataspecs,
        backbone_config=config.models.body_registry[config.models.use_body],
        conditioning=config.models.conditioning,
        enable_clamp=config.models.flags.enable_clamp,
        clamp_range=config.models.clamp_range
    )
    # build the session
    session_context=session.SessionBuildContext(
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
    # return the runner
    return runner.run
