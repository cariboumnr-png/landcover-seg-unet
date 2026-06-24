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
import landseg.adapters.api.configurators as configurators
import landseg.study.sweep.presets as presets

class StudySweepConfigurator(configurators.BaseConfigurator):
    '''Configure a study sweep session.'''

    def __init__(
        self,
        experiment_root: str,
        dataset_name: str,
        optuna_storage: str,
        seed: int = 42
    ):
        super().__init__(experiment_root, dataset_name, 'study-sweep')
        #
        self._cfg.pipeline.study_sweep.storage = optuna_storage
        self._cfg.pipeline.study_sweep.seed = seed

    # ----- sweep configs
    def set_sweep(
        self,
        study_name: str,
        preset_name: str,
        n_trials: int,
        direction: str = 'maximize',
    ) -> typing.Self:
        '''Set study sweep parameters.'''

        presets.resolve(preset_name)
        sweep_cfg = self._cfg.pipeline.study_sweep
        sweep_cfg.study_name = study_name
        sweep_cfg.preset_name = preset_name
        sweep_cfg.n_trials = n_trials
        sweep_cfg.direction = direction
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
