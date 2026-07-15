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
Configurator base class
'''

# standard imports
import typing
# local imports
import landseg.configs as configs

class BaseConfigurator:
    '''Configure data ingestion.'''

    def __init__(
        self,
        experiment_root: str,
        dataset_name: str,
        pipeline_name: str
    ):
        '''Initialize the configurator'''

        # init a default RootConfig instance
        self._cfg = configs.RootConfig()
        # set dataset name
        self._cfg.foundation.datablocks.name = dataset_name
        # set artifact output dirpaths
        self._cfg.execution.exp_root = experiment_root
        self._cfg.foundation.output_dpath = (
            f'{experiment_root}/artifacts/{dataset_name}/foundation'
        )
        self._cfg.transform.output_dpath = (
            f'{experiment_root}/artifacts/{dataset_name}/transform'
        )
        # set pipeline name
        self._cfg.pipeline.name = pipeline_name

    @property
    def running_root_config(self) -> configs.RootConfig:
        '''Validate and return the `RootConfig`,'''
        match self._cfg.pipeline.name:
            case 'data-ingest':
                self._cfg.foundation.validate()
            case 'data-prepare':
                self._cfg.transform.validate()
            case 'model-train':
                self._cfg.models.validate()
                self._cfg.session.validate()
            case 'study-sweep':
                self._cfg.models.validate()
                self._cfg.session.validate()
        return self._cfg

    # ----- shared methods for configuring runtime sessions
    def set_data_loading(
        self,
        batch_size: int,
        patch_size: int
    ) -> typing.Self:
        '''Set data sizes.'''
        self._cfg.session.data_loader.batch_size = batch_size
        self._cfg.session.data_loader.patch_size = patch_size
        return self

    def set_domain_source(
        self,
        category_domain: str | None,
        continuous_domain: str | None,
    ) -> typing.Self:
        '''Set data source'''
        self._cfg.dataspecs.domain_ids_name = category_domain
        self._cfg.dataspecs.domain_vec_name = continuous_domain
        return self

    def set_tasks(
        self,
        logit_adjust_alpha: float,
        exclude_classes: dict[str, list[int]] | None,
        loss_weights: dict[str, float],
    ) -> typing.Self:
        '''Set engine tasks.'''
        self._cfg.session.engine_exec.logit_adjust_alpha = logit_adjust_alpha
        self._cfg.session.engine_tasks.excluded_cls = exclude_classes
        losses = self._cfg.session.engine_tasks.loss_configs
        losses.focal.weight = loss_weights.get('focal', 0.5)
        losses.dice.weight = loss_weights.get('dice', 0.5)
        losses.spectral.weight = loss_weights.get('spectral', 1e-3)
        losses.tv.weight = loss_weights.get('total_var', 1e-4)
        return self

    def set_runtime(
        self,
        max_epochs: int,
        active_heads: list[str] | None,
        patience_epoch: int | None,
        track_heads: dict[str, float] | None,
    ) -> typing.Self:
        '''Set training runtime behaviour.'''
        orchestration = self._cfg.session.orchestration
        orchestration.curriculum.single.phases[0].num_epochs = max_epochs
        orchestration.curriculum.single.phases[0].active_heads = active_heads
        orchestration.monitor.patience = patience_epoch
        orchestration.monitor.track_heads = track_heads
        return self
