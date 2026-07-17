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

# standard imports
import datetime
import time
# local imports
import landseg._constants as c
import landseg.artifacts as artifacts
import landseg.configs as configs
import landseg.geopipe as geopipe
import landseg.models as models
import landseg.session as session


def train(config: configs.RootConfig) -> None:
    '''
    Run a full training job.

    Creates an run directory, builds `DataSpecs` from the prepared
    artifacts and schema, instantiates the model, and executes the runner.

    Args:
        config: RootConfig with model, trainer, and runner settings.
    '''

    def _cur_time():
        return datetime.datetime.now().strftime(c.TF_ISO8601)

    # init session results paths and create run io folder tree
    ss_paths = artifacts.ResultsPaths(f'{config.execution.exp_root}/results')
    ss_paths.init(config.session.orchestration.schedule.resume_from_last)

    # persist running config as JSON
    config_ctrl = artifacts.Controller[dict](ss_paths.config) # no policy
    config_ctrl.persist(config.as_dict)

    # parse verbosity
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

    # init a SessionLogger
    logger = session.SessionLogger(
        name='session',
        log_file=ss_paths.summary,
        console_lvl=console_level,
        enable_file_log=False
    )
    logger.init_summary(
        run_id=ss_paths.run_id,
        pipeline=config.pipeline.name,
        start_time=_cur_time()
    )
    assert logger.summary # typing

    try:
        logger.log_sep()

        # collect artifacts and build `DataSpecs`
        logger.log('INFO', '[START] Data specifications setup')
        start_time = time.perf_counter()
        artifact_paths = artifacts.ArtifactPaths(
            f'{config.execution.exp_root}/artifacts/'
            f'{config.foundation.datablocks.name}'
        )
        dataspecs = geopipe.build_dataspec(
            artifact_paths,
            mode='default',
            ids_domain_name=config.dataspecs.domain_ids_name,
            vec_domain_name=config.dataspecs.domain_vec_name,
            print_out=print_out
        )
        d_setup = time.perf_counter() - start_time
        logger.log('INFO', f'[COMPLETE] Data specs setup (D_{d_setup:.2f}s)')

        logger.log_sep()

        # setup the model
        logger.log('INFO', '[START] Model assembly')
        start_time = time.perf_counter()
        model = models.build_multihead_unet(
            patch_size=config.session.data_loader.patch_size,
            dataspecs=dataspecs,
            unet_backbone_config=config.models.unet_backbone_config,
            conditioning_config=config.models.conditioning_config,
            enable_clamp=config.models.numeric_safety.enable_clamp,
            clamp_range=config.models.numeric_safety.clamp_range
        )
        d_model = time.perf_counter() - start_time
        logger.log('INFO', f'[COMPLETE] Model assembly (D_{d_model:.2f}s)')

        logger.log_sep()

        # build the session
        # build session based on continuous or curriculum mode
        match config.session.mode:
            case 'continuous':
                session_context = session.SessionBuildContext(
                    device=c.DEVICE,
                    verbose_runner=print_out,
                    session_paths=ss_paths,
                )
                runner = session.factory.build_continous_training_session(
                    dataspecs=dataspecs,
                    model=model,
                    config=config.session,
                    context=session_context,
                    logger=logger
                )
            case 'curriculum':
                session_context = session.SessionBuildContext(
                    device=c.DEVICE,
                    verbose_runner=print_out,
                    session_paths=ss_paths,
                )
                runner = session.factory.build_curriculum_training_session(
                    dataspecs=dataspecs,
                    model=model,
                    config=config.session,
                    context=session_context,
                    logger=logger
                )
            case _:
                raise ValueError(f'Invalid training mode: {config.session.mode}')

        # run session execution
        logger.log('INFO', '[START] Training session')
        start_time = time.perf_counter()
        final = runner.execute()
        d_exec = time.perf_counter() - start_time
        logger.log('INFO', f'[COMPLETE] Training session (D_{d_exec:.2f}s)')

        # update summary
        logger.summary['completed_at'] = _cur_time()
        logger.set_summary_status('SUCCESS')
        logger.set_results({
            'best_value': final,
            'duration_sec': d_setup + d_model + d_exec
        })

    except Exception as e:
        logger.set_summary_status('FAILED')
        logger.log('ERROR', f'Training pipeline failed: {e}', exc_info=True)
        raise e

    # close logger
    finally:
        logger.log_sep()
        logger.close() # summary JSON will be persisted
