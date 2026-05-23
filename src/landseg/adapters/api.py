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
Programmatic API entry
'''

# standard imports
import typing
# local imports
import landseg.configs as configs
import landseg.execution as execution
import landseg.utils as utils

class DataIngestionConfigurator:
    '''Configure data ingestion.'''

    def __init__(
        self,
        experiment_root: str,
        dataset_name: str
    ):
        '''Initialize the configurator'''

        self._cfg = configs.RootConfig() # with all default values
        # set output dirpaths
        self._cfg.execution.exp_root = experiment_root
        self._cfg.foundation.output_dpath = (
            f'{experiment_root}/artifacts/{dataset_name}/foundation'
        )
        # here we set pipeline to data-ingest
        self._cfg.pipeline.name = 'data-ingest'

    @property
    def running_root_config(self) -> configs.RootConfig:
        '''Validate and return the `RootConfig`,'''
        self._cfg.foundation.validate()
        return self._cfg

    def set_grid(
        self,
        crs: str,
        reference_raster_fpath: str,
        tile_size: int,
        tile_overlap: int
    ):
        '''Set study extent and grid specs.'''
        self._cfg.foundation.grid.crs = crs
        self._cfg.foundation.grid.extent.filepath = reference_raster_fpath
        self._cfg.foundation.grid.tile_specs.size_row = tile_size
        self._cfg.foundation.grid.tile_specs.size_col = tile_size
        self._cfg.foundation.grid.tile_specs.overlap_row = tile_overlap
        self._cfg.foundation.grid.tile_specs.overlap_col = tile_overlap

    def set_domains(
        self,
        domains: list[tuple[str, int]]
    ):
        '''Set domain data source.'''
        for fpath, index_base in domains:
            self._cfg.foundation.domains.add_domain(fpath, index_base)

    def set_model_dev_data(
        self,
        model_dev_image: str,
        model_dev_label: str,
        data_config: str
    ):
        '''Set trainig data source.'''
        self._cfg.foundation.datablocks.filepaths.dev_image = model_dev_image
        self._cfg.foundation.datablocks.filepaths.dev_label = model_dev_label
        self._cfg.foundation.datablocks.filepaths.config = data_config

    def set_test_holdout_data(
        self,
        test_holdout_image: str,
        test_holdout_label: str
    ):
        '''Set test holdout data source.'''
        self._cfg.foundation.datablocks.filepaths.test_image = test_holdout_image
        self._cfg.foundation.datablocks.filepaths.test_label = test_holdout_label

class DataPreparationConfigurator:
    '''Configure data ingestion.'''

    def __init__(
        self,
        experiment_root: str,
        dataset_name: str
    ):
        '''Initialize the configurator'''

        self._cfg = configs.RootConfig() # with all default values
        # set output dirpaths
        self._cfg.execution.exp_root = experiment_root
        self._cfg.foundation.output_dpath = (
            f'{experiment_root}/artifacts/{dataset_name}/foundation'
        )
        self._cfg.transform.output_dpath = (
            f'{experiment_root}/artifacts/{dataset_name}/transform'
        )
        # here we set pipeline to data-prepare
        self._cfg.pipeline.name = 'data-prepare'

    @property
    def running_root_config(self) -> configs.RootConfig:
        '''Validate and return the `RootConfig`,'''
        self._cfg.transform.validate()
        return self._cfg

    def set_partition(
        self,
        validation_blocks_ratio: float,
        test_holdout_blocks_ratio: float
    ):
        '''Set block ratios for validation and test holdout.'''
        self._cfg.transform.partition.val_ratio = validation_blocks_ratio
        self._cfg.transform.partition.test_ratio = test_holdout_blocks_ratio

    def set_scoring(
        self,
        reward_classes: dict[int, float]
    ):
        '''Set block scoring criteria.'''
        self._cfg.transform.scoring.reward = reward_classes

    def set_hydration(self):
        '''Set blocks hydration'''

class TrainingSessionConfigurator:
    '''Configure a training session.'''

    def __init__(
        self,
        experiment_root: str,
        dataset_name: str
    ):
        '''Initialize the configurator'''

        self._cfg = configs.RootConfig() # with all default values
        # set experiment root
        self._cfg.execution.exp_root = experiment_root
        # set datablocks source
        self._cfg.foundation.datablocks.name = dataset_name
        # here we default to run a continuous training session
        self._cfg.pipeline.name = 'model-train'

    @property
    def running_root_config(self) -> configs.RootConfig:
        '''Validate and return the `RootConfig`,'''
        # here we are only validating settings related to the training
        self._cfg.models.validate()
        self._cfg.session.validate()
        return self._cfg

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
        category_domain: str,
        continuous_domain: str,
    ) -> typing.Self:
        '''Set data source'''
        self._cfg.dataspecs.domain_ids_name = category_domain
        self._cfg.dataspecs.domain_vec_name = continuous_domain
        return self

    def set_model(
        self,
        body: str,
        base_channel: int
    ) -> typing.Self:
        '''Set model body.'''
        self._cfg.models.model_body = body
        self._cfg.models.set_base_channel(base_channel)
        return self

    def set_optimization(
        self,
        optimizer: typing.Literal['AdamW'],
        learning_rate: float,
        weight_decay: float,
        scheduler: typing.Literal['CosAnneal']
    ) -> typing.Self:
        '''Set model optimization.'''
        engine_optim = self._cfg.session.engine_optim
        engine_optim.opt_cls = optimizer
        engine_optim.lr = learning_rate
        engine_optim.weight_decay = weight_decay
        engine_optim.sched_cls  = scheduler
        return self

    def set_objectives(
        self,
        focal_loss_weight: float,
        dice_loss_weight: float,
        spectral_loss_weight: float,
        tv_loss_weight: float
    ) -> typing.Self:
        '''Set loss weights.'''
        loss_types = self._cfg.session.engine_tasks.loss_types
        loss_types.focal.weight = focal_loss_weight
        loss_types.dice.weight = dice_loss_weight
        loss_types.spectral.weight = spectral_loss_weight
        loss_types.tv.weight = tv_loss_weight
        return self

    def set_runtime(
        self,
        max_epochs: int,
        patience_epoch: int | None,
        logit_adjust_alpha: float
    ) -> typing.Self:
        '''Set training runtime behaviour.'''
        orchestration = self._cfg.session.orchestration
        orchestration.curriculum.single.phases[0].num_epochs = max_epochs
        orchestration.monitor.patience = patience_epoch
        self._cfg.session.engine_exec.logit_adjust_alpha = logit_adjust_alpha
        return self

def run(root_config: configs.RootConfig):
    '''Run pipeline'''

    logger = utils.Logger('api', './api.log')
    try:
        logger.log('INFO', f'Running pipeline: {root_config.pipeline.name}')
        return execution.execute_pipeline(root_config)
    except KeyboardInterrupt:
        logger.log('INFO', 'Execution interrupted')
        raise
    except Exception:
        logger.log(
            'CRITICAL',
            'Unhandled exception occurred during API execution',
            exc_info=True,
        )
        raise
