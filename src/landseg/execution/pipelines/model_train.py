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
Training entrypoint.

Builds data specifications from produced artifacts, constructs the
model, and runs the multi-phase training runner.
'''

# local imports
import landseg._constants as c
import landseg.artifacts as artifacts
import landseg.configs as configs
import landseg.geopipe as geopipe
import landseg.models as models
import landseg.session as session
import landseg.utils as utils

def train(config: configs.RootConfig) -> session.SessionMetadata:
    '''
    Run a full training job.

    Creates an run directory, builds `DataSpecs` from the prepared
    artifacts and schema, instantiates the model, and executes the runner.

    Args:
        config: RootConfig with model, trainer, and runner settings.
    '''

    # init run io folder tree
    session_paths = artifacts.ResultsPaths(f'{config.execution.exp_root}/results')
    session_paths.init(trace_to_last=config.session.resume_from_last)

    # create the session metadata dict
    meta_ctrl = artifacts.Controller[dict](session_paths.meta)
    meta: session.SessionMetadata = {
        'status': 'running',
        'run_id': session_paths.run_id,
        'intent': 'training',
        'pipeline': config.pipeline.name,
        'created_at': session_paths.time(c.TF_ISO8601),
        'completed_at': None,
        'inputs': {},
        'summary': {}
    }
    meta_ctrl.persist(meta)

    # save running config per run
    config_ctrl = artifacts.Controller[dict](session_paths.config) # no policy
    config_ctrl.persist(config.as_dict)

    # verbosity
    match config.execution.verbosity:
        case 'full':
            console_level = 10
            print_out = True
        case 'select':
            console_level = 20
            print_out = True
        case 'silent':
            console_level = None
            print_out = False
        case _:
            raise ValueError(f'Invalid option: {config.execution.verbosity}')

    # create a centralized main logger
    logger = utils.Logger(
        name='main',
        log_file=session_paths.main_log_file,
        console_lvl=console_level
    )

    # collect artifacts and build dataspsec
    artifact_paths=artifacts.ArtifactPaths(f'{config.execution.exp_root}/artifacts')
    dataspecs = geopipe.build_dataspec(
        artifact_paths,
        mode='default',
        ids_domain_name=config.dataspecs.domain_ids_name,
        vec_domain_name=config.dataspecs.domain_vec_name,
        print_out=print_out
    )

    # setup the model
    model = models.build_multihead_unet(
        dataspecs=dataspecs,
        backbone_config=config.models.body_registry[config.models.use_body],
        conditioning=config.models.conditioning,
        enable_logit_adjust=config.models.flags.enable_logit_adjust,
        enable_clamp=config.models.flags.enable_clamp,
        clamp_range=config.models.clamp_range
    )

    # build the session
    session_context=session.SessionBuildContext(
        device=c.DEVICE,
        verbose_runner=print_out,
        session_paths=session_paths,
    )
    match config.session.mode:
        case 'continuous':
            runner = session.factory.build_continous_training_session(
                dataspecs=dataspecs,
                model=model,
                config=config.session,
                context=session_context,
                logger=logger
            )
        case 'curriculum':
            runner = session.factory.build_curriculum_training_session(
                dataspecs=dataspecs,
                model=model,
                config=config.session,
                context=session_context,
                logger=logger
            )
        case _:
            raise ValueError(f'Invalid training mode: {config.session.mode}')

    # run session in a block
    final = runner.execute()

    # close logger
    logger.close()

    # update metadata and return
    meta['completed_at'] = session_paths.time(c.TF_ISO8601)
    meta['summary'] = {}
    meta['summary']['best_value'] = final
    meta_ctrl.persist(meta)
    return meta
