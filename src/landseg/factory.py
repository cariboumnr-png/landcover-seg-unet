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
Project level factory.
'''

# third-party imports
import torch
# local imports
import landseg.configs as configs
import landseg.core as core
import landseg.core.ingest_protocols as ingest_protocols
import landseg.trainer_components as trainer_components
import landseg.trainer_engine as trainer_engine
import landseg.trainer_runner as trainer_runner
import landseg.utils as utils

# -------------------------------Public Function-------------------------------
def build_trainer(
    dataspecs: ingest_protocols.DataSpecs,
    model: core.MultiheadModelLike,
    config: configs.TrainerCfg,
    logger: utils.Logger,
    **kwargs
) -> trainer_engine.MultiHeadTrainer:
    '''
    Build a trainer from model, components, and config.

    Args:
        kwargs:
            runtime convenience flags
            - skip_log: bool
            - enable_train_la: bool
            - enable_val_la: bool
            - enable_test_la: bool
    '''

    # gather trainer components
    runtime_components = trainer_components.build_trainer_components(
        dataspecs,
        model,
        config,
        logger
    )

    # get runtime traine config
    runtime_config = trainer_engine.get_config(config.runtime)

    # get currently avalaible device
    available_device = 'cuda' if torch.cuda.is_available() else 'cpu'

    trainer = trainer_engine.MultiHeadTrainer(
        model,
        runtime_components,
        runtime_config,
        available_device,
        **kwargs
    )

    return trainer

def build_runner(
    experiment_dir: str,
    dataspecs: ingest_protocols.DataSpecs,
    model: core.MultiheadModelLike,
    config: configs.RootConfig,
    logger: utils.Logger,
    **kwargs
) -> trainer_runner.Runner:
    '''Setup training runner.'''

    # build trainer
    trainer = build_trainer(dataspecs, model, config.trainer, logger, **kwargs)

    # get phases
    phases = _generate_phases(config.runner)

    # build and return runner
    runner = trainer_runner.Runner(trainer, phases, experiment_dir, logger)
    return runner

# ------------------------------private  function------------------------------
def _generate_phases(config: configs.RunnerCfg) -> list[trainer_runner.Phase]:
    '''doc'''

    # config accesor
    phases: list[trainer_runner.Phase] = []
    # iterate through phases in config (1-based)
    for cfg in config.phases:
        phases.append(
            trainer_runner.Phase(
                name=cfg.name,
                num_epochs=cfg.num_epochs,
                heads=trainer_runner.HeadsConifg(
                    cfg.heads.active_heads,
                    cfg.heads.frozen_heads,
                    cfg.heads.masked_classes
                ),
                la_scheme=trainer_runner.LogitAdjustScheme(
                    cfg.logit_adjust.alpha,
                    cfg.logit_adjust.train,
                    cfg.logit_adjust.val,
                    cfg.logit_adjust.test,
                ),
                lr_scale=cfg.lr_scale
            )
        )

    # return
    return phases
