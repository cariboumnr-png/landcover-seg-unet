# =========================================================================== #
#            Copyright © His Majesty the King in right of Ontario,            #
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

'''Study sweep configurator.'''

# standard imports
import pathlib
import typing
# local imports
import landseg._constants as c
import landseg.adapters.api.configurators as configurators

class StudySweepConfigurator(configurators.BaseConfigurator):
    '''Configure a study sweep session.'''

    def __init__(
        self,
        *,
        upstream_dpath: str | pathlib.Path | None = None,
        output_dpath: str | pathlib.Path | None = None,
        experiment_root: str | pathlib.Path | None = None,
        optuna_storage: str | pathlib.Path | None = None,
        seed: int = 42,
    ):
        '''
        Initialize the study sweep configurator.

        Args:
            upstream_dpath:
                Optional directory path for upstream preparation
                outputs.
            output_dpath:
                Optional destination directory for sweep outputs.
            experiment_root:
                Optional experiment root directory for unified mode.
            optuna_storage:
                Optional Optuna storage URL or database path.
            seed:
                Random seed for hyperparameter sampling.
        '''
        super().__init__(
            'study-sweep',
            upstream_dpath=upstream_dpath,
            output_dpath=output_dpath,
            experiment_root=experiment_root,
        )

        self.study = self.cfg.study
        if optuna_storage is not None:
            self.cfg.study.sweep.storage = str(optuna_storage)
        elif self.cfg.session.output_dpath:
            self.cfg.study.sweep.storage = (
                f'sqlite:///{self.cfg.session.output_dpath}/optuna.db'
            )
        self.cfg.study.sweep.seed = seed

    # ----- sweep configs
    def set_sweep(
        self,
        study_name: str,
        preset_name: str,
        n_trials: int,
        direction: str = 'maximize',
    ) -> typing.Self:
        '''
        Set study sweep parameters.

        Args:
            study_name:
                Unique study name registered in Optuna storage.
            preset_name:
                Hyperparameter search preset identifier.
            n_trials:
                Number of optimization trials to execute.
            direction:
                Optimization direction (`'maximize'` or `'minimize'`).

        Returns:
            typing.Self:
                Configurator instance for method chaining.

        Raises:
            ValueError:
                If `preset_name` is not a recognized sweep preset.
        '''
        if preset_name not in c.SWEEP_PRESETS:
            raise ValueError(
                f'Unknown preset: {preset_name!r}; allowed: {c.SWEEP_PRESETS}'
            )
        sweep_cfg = self.cfg.study.sweep
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
        '''
        Set hyperparameter ranges for the base objectives preset.

        Args:
            learning_rate:
                Optional `(low, high)` bounds for learning rate search.
            batch_size:
                Optional `(low, high, step)` tuple for batch size.

        Returns:
            typing.Self:
                Configurator instance for method chaining.
        '''
        if learning_rate is not None:
            self.study.optimization.base.learning_rate = learning_rate
        if batch_size is not None:
            self.study.optimization.base.batch_size = batch_size
        return self

    def set_quick_preset_ranges(
        self,
        learning_rate: tuple[float, float] | None = None,
        weight_decay: tuple[float, float] | None = None,
        batch_size: tuple[int, int, int] | None = None,
        use_amp: list[bool] | None = None
    ) -> typing.Self:
        '''
        Set hyperparameter ranges for the quick sweep preset.

        Args:
            learning_rate:
                Optional `(low, high)` bounds for learning rate search.
            weight_decay:
                Optional `(low, high)` bounds for weight decay search.
            batch_size:
                Optional `(low, high, step)` tuple for batch size.
            use_amp:
                Optional list of mixed-precision toggle choices.

        Returns:
            typing.Self:
                Configurator instance for method chaining.
        '''
        if learning_rate is not None:
            self.study.optimization.optimizer.learning_rate = learning_rate
        if weight_decay is not None:
            self.study.optimization.optimizer.weight_decay = weight_decay
        if batch_size is not None:
            self.study.optimization.throughput.batch_size = batch_size
        if use_amp is not None:
            self.study.optimization.throughput.use_amp = use_amp
        return self

    def set_capacity_preset_ranges(
        self,
        *,
        model_body: list[str] | None = None,
        base_channel: tuple[int, int, int] | None = None,
        bottleneck: list[str] | None = None,
        num_conv_blks: tuple[int, int, int] | None = None,
        num_transformer_blks: tuple[int, int, int] | None = None,
        num_heads: list[int] | None = None,
        mlp_ratio: tuple[float, float] | None = None,
        dropout: tuple[float, float] | None = None,
        attn_dropout: tuple[float, float] | None = None
    ) -> typing.Self:
        '''
        Set hyperparameter ranges for capacity and architecture sweeps.

        Args:
            model_body:
                Optional candidate backbone identifiers.
            base_channel:
                Optional `(low, high, step)` tuple for base channels.
            bottleneck:
                Optional candidate bottleneck identifiers.
            num_conv_blks:
                Optional `(low, high, step)` tuple for conv block count.
            num_transformer_blks:
                Optional `(low, high, step)` tuple for transformer
                block count.
            num_heads:
                Optional candidate attention head counts.
            mlp_ratio:
                Optional `(low, high)` bounds for transformer MLP ratio.
            dropout:
                Optional `(low, high)` bounds for general dropout rate.
            attn_dropout:
                Optional `(low, high)` bounds for attention dropout.

        Returns:
            typing.Self:
                Configurator instance for method chaining.
        '''
        # architecture
        if model_body is not None:
            self.study.architecture.architecture.model_body = model_body
        if base_channel is not None:
            self.study.architecture.architecture.base_channel = base_channel
        # bottleneck
        if bottleneck is not None:
            self.study.architecture.architecture.bottleneck = bottleneck
            self.study.architecture.bottleneck.bottleneck = bottleneck
        if num_conv_blks is not None:
            self.study.architecture.bottleneck.num_conv_blocks = num_conv_blks
        if num_transformer_blks is not None:
            self.study.architecture.bottleneck.num_transformer_blocks = (
                num_transformer_blks
            )
        if num_heads is not None:
            self.study.architecture.bottleneck.num_heads = num_heads
        if mlp_ratio is not None:
            self.study.architecture.bottleneck.mlp_ratio = mlp_ratio
        if dropout is not None:
            self.study.architecture.bottleneck.dropout = dropout
        if attn_dropout is not None:
            self.study.architecture.bottleneck.attn_dropout = attn_dropout
        return self
