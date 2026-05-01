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

# pylint: disable=missing-function-docstring

'''Factory to build the session-owned components.'''

# standard imports
import dataclasses
import typing
# local imports
import landseg.core as core
import landseg.session.components.data as data
import landseg.session.components.optim as optim
import landseg.session.components.task as task
import landseg.utils as utils

# ---------------------------------Public Type---------------------------------
class ComponentsConfigLike(typing.Protocol):
    '''Shape of the components building configuration.'''
    @property
    def loader(self) -> data.LoaderConfig: ...
    @property
    def task(self) -> task.TaskConfig: ...
    @property
    def optimization(self) -> optim.OptimConfig: ...

# ------------------------------Public  Dataclass------------------------------
@dataclasses.dataclass
class SessionComponents:
    '''A simple container of the session components.'''
    dataloaders: data.DataLoaders
    headspecs: task.HeadSpecs
    headlosses: task.HeadLosses
    headmetrics: task.HeadMetrics
    optimization: optim.Optimization

# -------------------------------Public Function-------------------------------
def build_session_components(
    data_specs: core.DataSpecs,
    model: core.MultiheadModelLike,
    config: ComponentsConfigLike,
    *,
    logger: utils.Logger,
 ) -> SessionComponents:
    '''Builder session components from data shape and configs.'''

    # data loader
    data_loaders = data.build_dataloaders(
        data_specs,
        config.loader,
        logger=logger,
    )

    # task - heads specifications
    headspecs = task.build_headspecs(
        data_specs,
        alpha_fn=config.task.alpha_fn,
        en_beta=config.task.en_beta,
        excluded_cls=config.task.excluded_cls
    )
    # task - heads loss modules
    headlosses = task.build_headlosses(
        headspecs,
        config=config.task,
        ignore_index=data_specs.meta.label_specs.ignore_index,
        spectral_band_indices=data_specs.meta.image_specs.spec_channels
    )
    # task - heads metric modules
    headmetrics = task.build_headmetrics(
        headspecs,
        ignore_index=data_specs.meta.label_specs.ignore_index
    )

    # optimizer and scheduler
    optimization = optim.build_optimization(model, config.optimization)

    # collect components
    return SessionComponents(
        dataloaders=data_loaders,
        headspecs=headspecs,
        headlosses=headlosses,
        headmetrics=headmetrics,
        optimization=optimization,
    )
