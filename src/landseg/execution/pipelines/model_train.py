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
Training entrypoint.

Builds data specifications from produced artifacts, constructs the
model, and runs the multi-phase training runner.
'''

# standard imports
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
    artifact_paths = artifacts.ArtifactPaths.from_config(config)
    session_paths = artifact_paths.session
    session_paths.init(config.session.orchestration.resume_from_last)

    # persist running config as JSON
    config_ctrl = artifacts.Controller[dict](session_paths.config) # no policy
    config_ctrl.persist(config.as_dict)

    # init a SessionLogger
    logger = session.SessionLogger(
        name='session',
        log_file=session_paths.summary,
        console_lvl=config.execution.console_level,
        enable_file_log=False
    )
    logger.init_summary(
        run_id=session_paths.run_id,
        pipeline=config.pipeline.name,
    )

    try:
        logger.log_sep()

        # collect artifacts and build `DataSpecs`
        logger.log('INFO', '[START] Data specifications setup')
        start_t = time.perf_counter()
        dataspecs = geopipe.build_dataspec(
            artifact_paths,
            mode='default',
            ids_domain_name=config.data.specification.domain_ids_name,
            vec_domain_name=config.data.specification.domain_vec_name
        )
        d_setup = time.perf_counter() - start_t
        logger.log('INFO', f'[COMPLETE] Data specs setup (D_{d_setup:.2f}s)')

        for s in dataspecs.summary:
            logger.log('INFO', s)
        logger.log_sep()

        # setup the model
        logger.log('INFO', '[START] Model assembly')
        start_t = time.perf_counter()
        model = models.build_multihead_unet(
            patch_size=config.session.dataloader.patch_size,
            dataspecs=dataspecs,
            unet_backbone_config=config.models.unet_backbone_config,
            conditioning_config=config.models.conditioning_config,
            enable_clamp=config.models.numeric_safety.enable_clamp,
            clamp_range=config.models.numeric_safety.clamp_range
        )
        d_model = time.perf_counter() - start_t
        logger.log('INFO', f'[COMPLETE] Model assembly (D_{d_model:.2f}s)')

        logger.set_inputs(_summarize_inputs(config, model, dataspecs))
        logger.log_sep()

        # build the session runner
        runner = session.build_session_runner(
            dataspecs=dataspecs,
            model=model,
            config=config.session,
            context=session.SessionBuildContext(
                device=c.DEVICE,
                session_paths=session_paths,
                eval_dataset='val',
                logger=logger
            ),
            session_type=typing.cast(
                typing.Literal['continuous', 'curriculum'],
                config.session.mode
            ) # guaruanteed by root config validation,
        )

        # run session execution
        logger.log('INFO', '[START] Training session')
        start_t = time.perf_counter()
        final = runner.execute()
        d_exec = time.perf_counter() - start_t
        logger.log('INFO', f'[COMPLETE] Training session (D_{d_exec:.2f}s)')

        logger.set_summary_status('SUCCESS')
        logger.set_results(_summarize_results(final, d_setup, d_model, d_exec))

    except Exception as e:
        logger.set_summary_status('FAILED')
        logger.log('ERROR', f'Training pipeline failed: {e}', exc_info=True)
        raise e

    # close logger
    finally:
        logger.log_sep()
        logger.close() # summary JSON will be persisted


# ----- private helpers (no schema TEMPORARY)
def _summarize_inputs(
    config: configs.RootConfig,
    model: torch.nn.Module,
    dataspecs: core.DataSpecs
) -> dict[str, typing.Any]:
    '''Summarize pipeline run environment and model metadata inputs.'''
    total_p = sum(p.numel() for p in model.parameters())
    trainable_p = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {
        'system': {
            'device':c.DEVICE_NAME,
            'torch_version': torch.__version__
        },
        'model': {
            'backbone': 'unet',
            'total_parameters': total_p,
            'trainable_parameters': trainable_p,
            'heads': list(dataspecs.heads.class_counts.keys())
        },
        'data': {
            'patch_size': config.session.dataloader.patch_size
        },
        'dataspecs': dataspecs.to_dict()
    }


def _summarize_results(
    final: float,
    d_setup: float,
    d_model: float,
    d_exec: float,
) -> dict[str, typing.Any]:
    '''Summarize peak memory and log final results and metrics.'''
    process = psutil.Process()
    peak_cpu_mb = float(process.memory_info().rss / (1024 * 1024))
    peak_gpu_mb = 0.0
    if torch.cuda.is_available():
        peak_gpu_mb = float(torch.cuda.max_memory_allocated() / (1024 * 1024))

    return {
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
    }
