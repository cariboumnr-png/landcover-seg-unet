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
Evaluating a model.
'''

# standard imports
import datetime
import typing
# local imports
import landseg._constants as c
import landseg.artifacts as artifacts
import landseg.configs as configs
import landseg.geopipe as geopipe
import landseg.models as models
import landseg.session as session


def evaluate(config: configs.RootConfig):
    '''
    Run a single evaluation pass.

    Creates an run directory, builds `DataSpecs` from the prepared
    artifacts and schema, instantiates the model, and executes the runner.

    Args:
        config: RootConfig with model, trainer, and runner settings.
    '''
    # init run io folder tree
    session_paths = artifacts.ResultsPaths(f'{config.execution.exp_root}/results')
    session_paths.init()

    # parse evaluation pipeline configs
    eval_config = config.pipeline.model_evaluate
    assert eval_config.checkpoint
    if eval_config.split not in ('val', 'test'):
        raise ValueError(f"Invalid split: {eval_config.split}")
    split: typing.Literal['val', 'test'] = eval_config.split

    # save running config per run
    ctrl = artifacts.Controller[dict](session_paths.config) # no policy
    ctrl.persist(config.as_dict)

    # init a SessionLogger
    logger = session.SessionLogger(
        name='session',
        log_file=session_paths.summary,
        console_lvl=20,
        enable_file_log=False
    )
    logger.init_summary(
        run_id=session_paths.run_id,
        pipeline=config.pipeline.name,
        start_time=datetime.datetime.now().strftime(c.TF_ISO8601)
    )
    logger.set_inputs({
        'checkpoint': eval_config.checkpoint,
        'split': split
    })
    assert logger.summary # typing

    try:
        logger.log_sep()

        # collect artifacts and build `DataSpecs`
        artifact_paths = artifacts.ArtifactPaths(
            f'{config.execution.exp_root}/artifacts/'
            f'{config.foundation.datablocks.name}'
        )
        dataspecs = geopipe.build_dataspec(
            artifact_paths,
            mode='test_only' if split == 'test' else 'val_only',
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

        # load checkpoint with no optimizer nor scheduler
        artifacts.load_checkpoint(
            model=model,
            fpath=eval_config.checkpoint,
            map_device=c.DEVICE,
            optimizer=None,
            scheduler=None,
        )

        # build session runner
        session_context = session.SessionBuildContext(
            device=c.DEVICE,
            session_paths=session_paths,
        )
        runner = session.factory.build_evaluate_session(
            dataspecs=dataspecs,
            model=model,
            config=config.session,
            context=session_context,
            logger=logger
        )

        # evaluate
        evaluation_results = runner.run_epoch(0) # will always run
        assert evaluation_results.validation
        _metrics = evaluation_results.validation.head_metrics
        metrics = {h: m.as_dict for h, m in _metrics.items()}

        # persist the validation log as the current outputs
        output_ctrl = artifacts.Controller[dict](session_paths.evaluation)
        output_ctrl.persist(metrics)

        # update summary
        logger.summary['completed_at'] = datetime.datetime.now().strftime(c.TF_ISO8601)
        logger.set_summary_status('SUCCESS')
        logger.set_results({'final': evaluation_results.target_metrics})

    except Exception as e:
        logger.set_summary_status('FAILED')
        logger.log('ERROR', f'Evaluation pipeline failed: {e}', exc_info=True)
        raise e

    # close logger
    finally:
        logger.log_sep()
        logger.close() # summary dict will be persisted

    return evaluation_results.target_metrics
