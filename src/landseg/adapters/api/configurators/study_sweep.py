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
Study sweep configurator
'''

# standard imports
import typing
# local imports
import landseg.configs as configs
import landseg.study.sweep.presets as presets

class StudySweepConfigurator:
    '''Configure a study sweep session.'''

    def __init__(
        self,
        experiment_root: str,
        dataset_name: str
    ):
        '''Initialize the configurator'''

        self._cfg = configs.RootConfig() # with all default values
        self._cfg.foundation.datablocks.name = dataset_name # give name
        # set experiment root
        self._cfg.execution.exp_root = experiment_root
        # set datablocks source
        self._cfg.foundation.datablocks.name = dataset_name
        # set pipeline to study-sweep
        self._cfg.pipeline.name = 'study-sweep'

    @property
    def running_root_config(self) -> configs.RootConfig:
        '''Validate and return the `RootConfig`,'''
        self._cfg.models.validate()
        self._cfg.session.validate()
        return self._cfg

    # ----- essential/shared configs
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

    def set_runtime(
        self,
        max_epochs: int,
        active_heads: list[str] | None,
        track_heads: dict[str, float] | None,
        patience_epoch: int | None,
        logit_adjust_alpha: float
    ) -> typing.Self:
        '''Set training runtime behaviour.'''
        orchestration = self._cfg.session.orchestration
        orchestration.curriculum.single.phases[0].num_epochs = max_epochs
        orchestration.curriculum.single.phases[0].active_heads = active_heads
        orchestration.monitor.track_heads = track_heads
        orchestration.monitor.patience = patience_epoch
        self._cfg.session.engine_exec.logit_adjust_alpha = logit_adjust_alpha
        return self

    def set_model(
        self,
        body: str,
        bottleneck: str,
        base_channel: int
    ) -> typing.Self:
        '''Set model body.'''
        self._cfg.models.model_body = body
        self._cfg.models.bottleneck = bottleneck
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
        loss_types = self._cfg.session.engine_tasks.loss_configs
        loss_types.focal.weight = focal_loss_weight
        loss_types.dice.weight = dice_loss_weight
        loss_types.spectral.weight = spectral_loss_weight
        loss_types.tv.weight = tv_loss_weight
        return self

    # ----- sweep configs
    def set_sweep(
        self,
        study_name: str,
        preset_name: str,
        n_trials: int,
        storage: str = 'sqlite:///optuna.db',
        direction: str = 'maximize',
        seed: int = 42
    ) -> typing.Self:
        '''Set study sweep parameters.'''

        presets.resolve(preset_name)
        sweep_cfg = self._cfg.pipeline.study_sweep
        sweep_cfg.study_name = study_name
        sweep_cfg.preset_name = preset_name
        sweep_cfg.n_trials = n_trials
        sweep_cfg.storage = storage
        sweep_cfg.direction = direction
        sweep_cfg.seed = seed
        return self

    def set_base_preset_ranges(
        self,
        learning_rate: tuple[float, float] | None = None,
        batch_size: tuple[int, int, int] | None = None
    ) -> typing.Self:
        '''Set ranges for base objectives preset.'''
        if learning_rate is not None:
            self._cfg.study.base.learning_rate = learning_rate
        if batch_size is not None:
            self._cfg.study.base.batch_size = batch_size
        return self

    def set_quick_preset_ranges(
        self,
        learning_rate: tuple[float, float] | None = None,
        weight_decay: tuple[float, float] | None = None,
        batch_size: tuple[int, int, int] | None = None,
        use_amp: list[bool] | None = None
    ) -> typing.Self:
        '''Set ranges for entry-level (quick) sweep preset.'''
        if learning_rate is not None:
            self._cfg.study.optimizer.learning_rate = learning_rate
        if weight_decay is not None:
            self._cfg.study.optimizer.weight_decay = weight_decay
        if batch_size is not None:
            self._cfg.study.throughput.batch_size = batch_size
        if use_amp is not None:
            self._cfg.study.throughput.use_amp = use_amp
        return self

    def set_capacity_preset_ranges(
        self,
        model_body: list[str] | None = None,
        base_channel: tuple[int, int, int] | None = None,
        bottleneck: list[str] | None = None,
        num_conv_blocks: tuple[int, int, int] | None = None,
        num_transformer_blocks: tuple[int, int, int] | None = None,
        num_heads: list[int] | None = None,
        mlp_ratio: tuple[float, float] | None = None,
        dropout: tuple[float, float] | None = None,
        attn_dropout: tuple[float, float] | None = None
    ) -> typing.Self:
        '''Set ranges for moderate-level (capacity/architecture) sweep preset.'''
        if model_body is not None:
            self._cfg.study.architecture.model_body = model_body
        if base_channel is not None:
            self._cfg.study.architecture.base_channel = base_channel
        if bottleneck is not None:
            self._cfg.study.architecture.bottleneck = bottleneck
            self._cfg.study.bottleneck.bottleneck = bottleneck
        if num_conv_blocks is not None:
            self._cfg.study.bottleneck.num_conv_blocks = num_conv_blocks
        if num_transformer_blocks is not None:
            self._cfg.study.bottleneck.num_transformer_blocks = num_transformer_blocks
        if num_heads is not None:
            self._cfg.study.bottleneck.num_heads = num_heads
        if mlp_ratio is not None:
            self._cfg.study.bottleneck.mlp_ratio = mlp_ratio
        if dropout is not None:
            self._cfg.study.bottleneck.dropout = dropout
        if attn_dropout is not None:
            self._cfg.study.bottleneck.attn_dropout = attn_dropout
        return self

    def set_mtl_quality_preset_ranges(
        self,
        focal_weight: tuple[float, float] | None = None,
        dice_weight: tuple[float, float] | None = None,
        spectral_weight: tuple[float, float] | None = None,
        tv_weight: tuple[float, float] | None = None,
        consistency_lambda: tuple[float, float] | None = None
    ) -> typing.Self:
        '''Set ranges for multitasking quality sweep preset.'''
        if focal_weight is not None:
            self._cfg.study.loss_balance.focal_weight = focal_weight
        if dice_weight is not None:
            self._cfg.study.loss_balance.dice_weight = dice_weight
        if spectral_weight is not None:
            self._cfg.study.loss_auxiliary.spectral_weight = spectral_weight
        if tv_weight is not None:
            self._cfg.study.loss_auxiliary.tv_weight = tv_weight
        if consistency_lambda is not None:
            self._cfg.study.regularization.consistency_lambda = consistency_lambda
        return self
