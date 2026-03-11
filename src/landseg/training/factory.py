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

'''Pipeline to build the trainer class.'''

# third-party imports
import torch
# local imports
import landseg.core as core
import landseg.configs as configs
import landseg.models as models
import landseg.training.callback as callback
import landseg.training.dataloading as dataloading
import landseg.training.heads as heads
import landseg.training.loss as loss
import landseg.training.metrics as metrics
import landseg.training.optim as optim
import landseg.training.trainer as trainer
import landseg.utils as utils

# -------------------------------Public Function-------------------------------
def build_trainer(
    data_specs: core.DataSpecs,
    datablock_cls: type[core.DataBlockLike],
    model_config: configs.ModelsCfg,
    trainer_config: configs.TrainerCfg,
    logger: utils.Logger,
    **kwargs
 ) -> trainer.MultiHeadTrainer:
    '''Builder trainer.'''

    # setup the model
    model = models.build_multihead_unet(
        body=trainer_config.model_body,
        dataspecs=data_specs,
        config=model_config
    )

    # compile data loaders
    data_loaders = dataloading.get_dataloaders(
        data_specs=data_specs,
        config=trainer_config.loader,
        datablock_cls=datablock_cls,
        logger=logger
    )

    # compile training heads basic specifications
    headspecs = heads.build_headspecs(
        data=data_specs,
        config=trainer_config.loss,
    )

    # compile training heads loss compute modules
    headlosses = loss.build_headlosses(
        headspecs=headspecs,
        config=trainer_config.loss,
        ignore_index=data_specs.meta.ignore_index,
    )

    # compile training heads metric compute modules
    headmetrics = metrics.build_headmetrics(
        headspecs=headspecs,
        ignore_index=data_specs.meta.ignore_index
    )

    # build optimizer and scheduler
    optimization = optim.build_optimization(
        model=model,
        config=trainer_config.optim
    )

    # generate callback instances
    callbacks = callback.build_callbacks(logger)

    # collect components
    trainer_components = trainer.TrainerComponents(
        model=model,
        dataloaders=data_loaders,
        headspecs=headspecs,
        headlosses=headlosses,
        headmetrics=headmetrics,
        optimization=optimization,
        callbacks=callbacks,
    )

    # generate runtime config dataclass from config
    trainer_runtime_config = trainer.get_config(trainer_config.runtime)

    # get currently avalaible device
    available_device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # build and return a trainer class
    return trainer.MultiHeadTrainer(
        trainer_components,
        trainer_runtime_config,
        available_device,
        **kwargs
    )
