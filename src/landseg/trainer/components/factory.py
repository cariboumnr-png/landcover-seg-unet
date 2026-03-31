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

'''Factory to build the trainer components.'''

# standard imports
import dataclasses
# local imports
import landseg.configs as configs
import landseg.core as core
import landseg.trainer.components as components
import landseg.utils as utils

# ------------------------------Public  Dataclass------------------------------
@dataclasses.dataclass
class TrainerComponents:
    '''Collection of trainer components.'''
    callbacks: components.CallbackSet
    dataloaders: components.DataLoaders
    headspecs: components.HeadSpecs
    headlosses: components.HeadLosses
    headmetrics: components.HeadMetrics
    optimization: components.Optimization

# -------------------------------Public Function-------------------------------
def build_trainer_components(
    data_specs: core.DataSpecs,
    model: core.MultiheadModelLike,
    config: configs.TrainerCfg,
    logger: utils.Logger,
 ) -> TrainerComponents:
    '''Builder trainer.'''

    # generate callback instances
    callbacks = components.build_callbacks(logger)

    # compile data loaders
    data_loaders = components.build_dataloaders(
        data_specs,
        config.loader.batch_size,
        config.loader.patch_size,
        logger
    )

    # compile training heads basic specifications
    headspecs = components.build_headspecs(
        data_specs,
        config.loss.alpha_fn,
        en_beta=config.loss.en_beta
    )

    # compile training heads loss compute modules
    headlosses = components.build_headlosses(
        headspecs,
        config.loss.types,
        data_specs.meta.ignore_index,
    )

    # compile training heads metric compute modules
    headmetrics = components.build_headmetrics(
        headspecs,
        data_specs.meta.ignore_index
    )

    # build optimizer and scheduler
    optimization = components.build_optimization(
        model,
        config.optim.opt_cls,
        config.optim.lr,
        config.optim.weight_decay,
        config.optim.sched_cls,
        **config.optim.sched_args
    )

    # collect components
    return TrainerComponents(
        dataloaders=data_loaders,
        headspecs=headspecs,
        headlosses=headlosses,
        headmetrics=headmetrics,
        optimization=optimization,
        callbacks=callbacks,
    )
