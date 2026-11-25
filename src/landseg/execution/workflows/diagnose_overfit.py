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
Overfit test pipeline.

Delegates single-block diagnostic `DataSpecs` construction to
`landseg.geopipe` and trains until near-perfect IoU to validate the
end-to-end model and training stack.
'''

# standard imports
import typing
# local imports
import landseg._constants as c
import landseg.artifacts as artifacts
import landseg.configs as configs
import landseg.geopipe as geopipe
import landseg.models as models
import landseg.runtime as runtime
import landseg.session as session


# ----- overfit test pipeline
def execute_diagnose_overfit(config: configs.RootConfig) -> None:
    '''
    Run an overfit test on a single block.

    Builds or loads a diagnostic `DataSpecs` via `landseg.geopipe`,
    instantiates the model and trainer with minimal logging, and trains
    until an IoU threshold or the epoch limit is reached.

    Args:
        config: RootConfig with model/trainer settings.
    '''
    artifact_paths = artifacts.ArtifactPaths.from_config(config)
    root = artifact_paths.session.root / 'overfit_test'
    log_file = root / 'log' / 'overfit_summary.json'
    logger = session.SessionLogger('overfit', log_file=log_file)
    logger.init_summary(run_id='overfit_test', command='diagnose_overfit')

    try:
        logger.log_sep()
        logger.set_inputs({
            'exp_root': config.execution.exp_root,
            'patch_size': config.session.dataloader.patch_size,
            'lr': config.session.engine_optim.lr,
            'max_epoch': c.OVERFIT_MAX_EPOCH,
        })

        dataspecs = geopipe.build_overfit_dataspec(
            save_dpath=root,
            harmonization_paths=artifact_paths.data_harmonization,
            ingestion_paths=artifact_paths.data_ingestion,
            dem_pad_px=config.data.ingestion.datablocks.image_dem_pad,
            active_heads=(
                config.session.orchestration.single_phase.active_heads
            ),
            knowledge_root=config.execution.knowledge_dir,
            logger=logger,
        )

        model = models.build_multihead_unet(
            patch_size=config.session.dataloader.patch_size,
            dataspecs=dataspecs,
            unet_backbone_config=config.models.unet_backbone_config,
            conditioning_config=config.models.conditioning_config,
            enable_clamp=config.models.numeric_safety.enable_clamp,
            clamp_range=config.models.numeric_safety.clamp_range,
        )

        runner = session.build_session_runner(
            dataspecs=dataspecs,
            model=model,
            config=config.session,
            context=session.SessionBuildContext(
                device=runtime.resolve_device(config.session.device),
                session_paths=artifact_paths.session,  # no ops
                eval_dataset='val',
                logger=logger,
            ),
            session_type='overfit',
        )
        monitor_head = config.session.orchestration.monitor.track_heads
        active_heads = (
            list(monitor_head.keys())
            if monitor_head
            else list(dataspecs.heads.class_counts.keys())
        )
        runner.set_head_state(active_heads=active_heads)

        results = _run_overfit_loop(runner, config, logger)
        status = 'SUCCESS' if results['overfit_reached'] else 'FAILED'

        logger.set_results(results)
        logger.set_status(status)

    except Exception as e:
        logger.set_status('FAILED')
        logger.error(f'Overfit pipeline failed: {e}', exc_info=True)
        raise e

    finally:
        logger.log_sep()
        logger.close()


# ----- overfit epoch training loop helper
def _run_overfit_loop(
    runner: session.EpochRunner,
    config: configs.RootConfig,
    logger: session.SessionLogger,
) -> dict[str, typing.Any]:
    '''Execute epoch training loop until threshold or max epochs.'''
    max_epoch = c.OVERFIT_MAX_EPOCH
    lr = config.session.engine_optim.lr
    logger.info('Starting overfit test')
    logger.info(f'Maximum epoch: {max_epoch}')
    logger.info(f'Learning rate: {lr}')

    loss, iou = 0.0, 0.0
    for ep in range(1, max_epoch + 1):
        results = runner.run_epoch(ep)
        assert results.training and results.validation  # typing
        results.track(
            config.session.orchestration.monitor.metric_name,
            config.session.orchestration.monitor.track_heads,
        )
        loss = results.training.total_objective
        iou = results.target_metrics
        logger.info(f'Epoch: {ep:04d} | Loss: {loss:.4f} | IoU: {iou:.4f}')
        if iou >= 0.99:
            logger.info('Overfit reached - test complete')
            return {
                'final_epoch': ep,
                'final_loss': loss,
                'final_iou': iou,
                'overfit_reached': True,
            }

    logger.warning(f'IoU did not reach 99% after {max_epoch} epochs.')
    return {
        'final_epoch': max_epoch,
        'final_loss': loss,
        'final_iou': iou,
        'overfit_reached': False,
    }
