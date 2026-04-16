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
import dataclasses
# third-party imports
import torch
# local imports
import landseg.artifacts as artifacts
import landseg.configs as configs
import landseg.geopipe as geopipe
import landseg.models as models
import landseg.session as session
import landseg.utils as utils

def train(config: configs.RootConfig):
    '''
    Run a full training job.

    Creates an run directory, builds `DataSpecs` from the prepared
    artifacts and schema, instantiates the model, and executes the runner.

    Args:
        config: RootConfig with model, trainer, and runner settings.
    '''

    # init run io folder tree
    run_paths = artifacts.ResultsPaths(f'{config.exp_root}/results')
    run_paths.init()

    # save running config per run
    ctrl = artifacts.Controller[dict](run_paths.config) # generic, no policy
    ctrl.persist(dataclasses.asdict(config))

    # create a centralized main logger
    logger = utils.Logger('main', run_paths.main_log_file)

    # collect artifacts and build dataspsec
    artifact_paths=artifacts.ArtifactPaths(f'{config.exp_root}/artifacts')
    dataspecs = geopipe.build_dataspec(
        artifact_paths,
        ids_domain_name=config.trainer.runtime.data.domain_ids_name,
        vec_domain_name=config.trainer.runtime.data.domain_vec_name,
        print_out=True
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

    # trainer components
    components = session.build_session_components(
        data_specs=dataspecs,
        model=model,
        data_config=config.trainer.loader,
        task_config=config.trainer.loss,
        optim_config=config.trainer.optimization,
        logger=logger,
    )
    # shared runtime state
    state = session.init_state(
        components,
        use_amp=config.trainer.runtime.precision.use_amp,
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )
    # set callback up
    for c in components.callbacks:
        c.setup(
            state,
            config.trainer.runtime,
            device='cuda' if torch.cuda.is_available() else 'cpu',
            skip_log=False
        )
    # batch engine
    engine = session.BatchExecutionEngine(
        model=model,
        state=state,
        parent_map={k: v.parent_head for k, v in components.headspecs.as_dict().items()},
        use_amp=config.trainer.runtime.precision.use_amp,
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )
    # trainer
    trainer = session.MultiHeadTrainer(
        engine=engine,
        state=state,
        components=components,
        config=config.trainer.runtime,
        device='cuda' if torch.cuda.is_available() else 'cpu',
        use_amp=config.trainer.runtime.precision.use_amp,
        grad_clip_norm=config.trainer.runtime.optimization.grad_clip_norm,
        log_every=config.trainer.runtime.schedule.log_every,
    )
    # evaluator
    evaluator = session.MultiHeadEvaluator(
        engine=engine,
        state=state,
        components=components,
        config=config.trainer.runtime,
        device='cuda' if torch.cuda.is_available() else 'cpu',
        track_mode=config.trainer.runtime.monitor.track_mode,
        track_head_name=config.trainer.runtime.monitor.track_head_name,
        min_delta=config.trainer.runtime.schedule.min_delta
    )
    # get phases
    phases = [
        session.Phase(
            name=cfg.name,
            num_epochs=cfg.num_epochs,
            heads=cfg.heads,
            logit_adjust=cfg.logit_adjust,
            lr_scale=cfg.lr_scale,
            finished=False
        ) for cfg in config.runner.phases
    ]

    # build controller and run
    runner = session.Runner(
        trainer=trainer,
        evaluator=evaluator,
        schedule=config.trainer.runtime.schedule,
        phases=phases,
        run_paths=run_paths,
        logger=logger
    )
    runner.fit()
