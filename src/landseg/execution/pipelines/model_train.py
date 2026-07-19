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
import typing
# third-party imports
import psutil
import torch
# local imports
import landseg._constants as c
import landseg.artifacts as artifacts
import landseg.configs as configs
import landseg.core as core
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
    # init session results paths and create run io folder tree
    ss_paths = artifacts.ResultsPaths(f'{config.execution.exp_root}/results')
    ss_paths.init(config.session.orchestration.schedule.resume_from_last)

    # persist running config as JSON
    config_ctrl = artifacts.Controller[dict](ss_paths.config) # no policy
    config_ctrl.persist(config.as_dict)

    # parse verbosity
    console_level = _parse_verbosity(config.execution.verbosity)

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
        start_time=_current_time()
    )
    assert logger.summary # typing

    try:
        logger.log_sep()

        # collect artifacts and build `DataSpecs`
        logger.log('INFO', '[START] Data specifications setup')
        start_t = time.perf_counter()
        artifact_paths = artifacts.ArtifactPaths(
            f'{config.execution.exp_root}/artifacts/'
            f'{config.foundation.datablocks.name}'
        )
        dataspecs = geopipe.build_dataspec(
            artifact_paths,
            mode='default',
            ids_domain_name=config.dataspecs.domain_ids_name,
            vec_domain_name=config.dataspecs.domain_vec_name
        )
        d_setup = time.perf_counter() - start_t
        logger.log('INFO', f'[COMPLETE] Data specs setup (D_{d_setup:.2f}s)')

        # log a clean summary of dataspecs to console when verbose
        _log_dataspecs_summary(logger, dataspecs, console_level)

        logger.log_sep()

        # setup the model
        logger.log('INFO', '[START] Model assembly')
        start_t = time.perf_counter()
        model = models.build_multihead_unet(
            patch_size=config.session.data_loader.patch_size,
            dataspecs=dataspecs,
            unet_backbone_config=config.models.unet_backbone_config,
            conditioning_config=config.models.conditioning_config,
            enable_clamp=config.models.numeric_safety.enable_clamp,
            clamp_range=config.models.numeric_safety.clamp_range
        )
        d_model = time.perf_counter() - start_t
        logger.log('INFO', f'[COMPLETE] Model assembly (D_{d_model:.2f}s)')

        _log_inputs(logger, config, model, dataspecs)

        logger.log_sep()

        # build the session runner
        runner = _build_session_runner(
            config,
            dataspecs,
            model,
            ss_paths,
            logger
        )

        # run session execution
        logger.log('INFO', '[START] Training session')
        start_t = time.perf_counter()
        final = runner.execute()
        d_exec = time.perf_counter() - start_t
        logger.log('INFO', f'[COMPLETE] Training session (D_{d_exec:.2f}s)')

        _log_results(logger, final, d_setup, d_model, d_exec)

    except Exception as e:
        logger.set_summary_status('FAILED')
        logger.log('ERROR', f'Training pipeline failed: {e}', exc_info=True)
        raise e

    # close logger
    finally:
        logger.log_sep()
        logger.close() # summary JSON will be persisted


def _parse_verbosity(verbosity: str) -> int | None:
    '''Parse verbosity option into console level.'''
    match verbosity:
        case 'full':
            return 10
        case 'select':
            return 20
        case 'silent':
            return None
        case _:
            raise ValueError(f'Invalid option: {verbosity}')


def _get_device_name() -> str:
    '''Retrieve execution device hardware name.'''
    if c.DEVICE.startswith('cuda'):
        if torch.cuda.is_available():
            return torch.cuda.get_device_name(0)
        return 'cuda (unavailable)'
    return c.DEVICE


def _log_inputs(
    logger: session.SessionLogger,
    config: configs.RootConfig,
    model: torch.nn.Module,
    dataspecs: core.DataSpecs
) -> None:
    '''Log pipeline run environment and model metadata inputs.'''
    total_p = sum(p.numel() for p in model.parameters())
    trainable_p = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.set_inputs({
        'system': {
            'device': _get_device_name(),
            'torch_version': torch.__version__
        },
        'model': {
            'backbone': 'unet',
            'total_parameters': total_p,
            'trainable_parameters': trainable_p,
            'heads': list(dataspecs.heads.class_counts.keys())
        },
        'data': {
            'dataset_name': config.foundation.datablocks.name,
            'patch_size': config.session.data_loader.patch_size
        },
        'dataspecs': dataspecs.to_dict()
    })


def _build_session_runner(
    config: configs.RootConfig,
    dataspecs: core.DataSpecs,
    model: core.MultiheadModelLike,
    ss_paths: artifacts.ResultsPaths,
    logger: session.SessionLogger
) -> typing.Any:
    '''Build the session runner based on mode.'''
    match config.session.mode:
        case 'continuous':
            session_context = session.SessionBuildContext(
                device=c.DEVICE,
                session_paths=ss_paths,
            )
            return session.factory.build_continous_training_session(
                dataspecs=dataspecs,
                model=model,
                config=config.session,
                context=session_context,
                logger=logger
            )
        case 'curriculum':
            session_context = session.SessionBuildContext(
                device=c.DEVICE,
                session_paths=ss_paths,
            )
            return session.factory.build_curriculum_training_session(
                dataspecs=dataspecs,
                model=model,
                config=config.session,
                context=session_context,
                logger=logger
            )
        case _:
            raise ValueError(f'Invalid training mode: {config.session.mode}')


def _log_results(
    logger: session.SessionLogger,
    final: float,
    d_setup: float,
    d_model: float,
    d_exec: float,
) -> None:
    '''Calculate peak memory and log final results and metrics.'''
    process = psutil.Process()
    peak_cpu_mb = float(process.memory_info().rss / (1024 * 1024))
    peak_gpu_mb = 0.0
    if torch.cuda.is_available():
        peak_gpu_mb = float(torch.cuda.max_memory_allocated() / (1024 * 1024))

    assert logger.summary is not None
    logger.summary['completed_at'] = _current_time()
    logger.set_summary_status('SUCCESS')
    logger.set_results({
        'best_value': final,
        'duration_sec': d_setup + d_model + d_exec,
        'durations': {
            'data_specs_setup_sec': d_setup,
            'model_assembly_sec': d_model,
            'execution_sec': d_exec
        },
        'system': {
            'peak_cpu_memory_mb': peak_cpu_mb,
            'peak_gpu_memory_mb': peak_gpu_mb
        }
    })


def _log_dataspecs_summary(
    logger: session.SessionLogger,
    dataspecs: geopipe.core.DataSpecs,
    console_level: int | None
) -> None:
    '''Log a concise, human-readable summary of the dataset specifications.'''
    if console_level is not None:
        img_ch = dataspecs.meta.image_specs.num_channels
        img_hw = dataspecs.meta.image_specs.height_width
        heads_str = ', '.join(dataspecs.heads.class_counts.keys())
        train_n = len(dataspecs.splits.train)
        val_n = len(dataspecs.splits.val)
        test_n = len(dataspecs.splits.test or {})

        logger.log(
            'INFO',
            f'Dataset name:\t{dataspecs.name} (mode: {dataspecs.mode})'
        )
        logger.log(
            'INFO',
            f'Image size:\t{img_ch} channels | {img_hw}x{img_hw}'
        )
        logger.log(
            'INFO',
            f'Target heads:\t{heads_str}'
        )
        logger.log(
            'INFO',
            f'Data splits:\t{train_n} train | {val_n} val | '
            f'{test_n} test blocks'
        )

def _current_time() -> str:
    '''Return current time as a formatted string.'''
    return datetime.datetime.now().strftime(c.TF_ISO8601)
