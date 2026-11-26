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

'''
Hyperparameter sweep trial configuration mutator.

Public APIs:
    - `TrialMutator`: Applies sampled hyperparameters to `RootConfig`.
'''

# standard imports
import typing
# local imports
import landseg.configs as configs


class TrialMutator: # pylint: disable=too-many-public-methods
    '''Mutates `RootConfig` fields for hyperparameter sweep trials.'''

    def __init__(self, config: configs.RootConfig) -> None:
        '''
        Initialize mutator with a trial root configuration instance.

        Args:
            config:
                Mutable root configuration updated by sweep presets.
        '''
        self.config = config

    @property
    def study(self) -> configs.StudyConfig:
        '''Return study configuration search spaces.'''
        return self.config.study

    # ----- data geometry
    def set_data_patch_size(self, patch_size: int) -> None:
        '''
        Set dataloader spatial patch size.

        Args:
            patch_size:
                Square patch side length in pixels.
        '''
        self.config.session.dataloader.patch_size = patch_size

    def set_data_batch_size(self, batch_size: int) -> None:
        '''
        Set dataloader mini-batch size.

        Args:
            batch_size:
                Number of samples per training batch.
        '''
        self.config.session.dataloader.batch_size = batch_size

    # ----- runtime optimization
    def set_optimizer_lr(self, lr: float) -> None:
        '''
        Set engine optimizer learning rate.

        Args:
            lr:
                Base optimizer learning rate.
        '''
        self.config.session.engine_optim.lr = lr

    def set_optimizer_weight_decay(self, weight_decay: float) -> None:
        '''
        Set engine optimizer weight decay coefficient.

        Args:
            weight_decay:
                L2 regularization penalty factor.
        '''
        self.config.session.engine_optim.weight_decay = weight_decay

    def set_optimizer_type(self, opt_cls: str) -> None:
        '''
        Set engine optimizer class identifier.

        Args:
            opt_cls:
                Optimizer registry name or class identifier.
        '''
        self.config.session.engine_optim.opt_cls = opt_cls

    def set_optimizer_scheduler_type(self, sched_cls: str | None) -> None:
        '''
        Set engine learning rate scheduler class identifier.

        Args:
            sched_cls:
                Scheduler identifier, or `None` to disable scheduling.
        '''
        self.config.session.engine_optim.sched_cls = sched_cls

    def set_optimizer_scheduler_args(
        self,
        sched_args: dict[str, typing.Any],
    ) -> None:
        '''
        Set engine learning rate scheduler keyword arguments.

        Args:
            sched_args:
                Keyword arguments passed to the scheduler constructor.
        '''
        self.config.session.engine_optim.sched_args = sched_args

    def set_grad_clip_norm(self, grad_clip_norm: float | None) -> None:
        '''
        Set engine gradient clipping maximum norm.

        Args:
            grad_clip_norm:
                Maximum L2 gradient norm, or `None` to disable clipping.
        '''
        self.config.session.engine_optim.grad_clip_norm = grad_clip_norm

    def set_runtime_use_amp(self, use_amp: bool) -> None:
        '''
        Set engine automatic mixed precision flag.

        Args:
            use_amp:
                Whether to enable mixed-precision execution.
        '''
        self.config.session.engine_exec.use_amp = use_amp

    def set_runtime_logit_adjust_alpha(self, alpha: float) -> None:
        '''
        Set engine logit adjustment scaling parameter.

        Args:
            alpha:
                Class-prior logit adjustment exponent.
        '''
        self.config.session.engine_exec.logit_adjust_alpha = alpha

    # ----- objective (loss + regularization)
    def set_objective_focal_weight(self, weight: float) -> None:
        '''
        Set focal loss component weight.

        Args:
            weight:
                Scalar multiplier applied to focal loss.
        '''
        self.config.session.engine_tasks.loss_configs.focal.weight = weight

    def set_objective_dice_weight(self, weight: float) -> None:
        '''
        Set dice loss component weight.

        Args:
            weight:
                Scalar multiplier applied to dice loss.
        '''
        self.config.session.engine_tasks.loss_configs.dice.weight = weight

    def set_objective_spectral_weight(self, weight: float) -> None:
        '''
        Set spectral loss component weight.

        Args:
            weight:
                Scalar multiplier applied to spectral consistency loss.
        '''
        self.config.session.engine_tasks.loss_configs.spectral.weight = weight

    def set_objective_tv_weight(self, weight: float) -> None:
        '''
        Set total variation loss component weight.

        Args:
            weight:
                Scalar multiplier applied to total variation loss.
        '''
        self.config.session.engine_tasks.loss_configs.tv.weight = weight

    def set_objective_ecological_weight(self, weight: float) -> None:
        '''
        Set ecological loss component weight.

        Args:
            weight:
                Scalar multiplier applied to ecological constraint loss.
        '''
        loss_cfg = self.config.session.engine_tasks.loss_configs
        loss_cfg.ecological.weight = weight

    # ----- multitask
    def set_mtl_consistency_lambda(self, value: float) -> None:
        '''
        Set multi-task consistency regularization lambda.

        Args:
            value:
                Scalar weight for cross-head consistency penalty.
        '''
        reg_config = self.config.session.engine_tasks.mtl_reg_configs
        reg_config.consistency_lambda = value

    def set_mtl_consistency_reduction(self, reduction: str) -> None:
        '''
        Set multi-task consistency loss reduction strategy.

        Args:
            reduction:
                Reduction mode name such as `'mean'` or `'sum'`.
        '''
        reg_config = self.config.session.engine_tasks.mtl_reg_configs
        reg_config.consistency_reduction = reduction

    # ----- architecture
    def set_model_body(self, model_body: str) -> None:
        '''
        Set UNet backbone model body identifier.

        Args:
            model_body:
                Registered model body variant name.
        '''
        self.config.models.model_body = model_body

    def set_model_base_channel(self, base_channel: int) -> None:
        '''
        Set base channel count on the active model body.

        Args:
            base_channel:
                Number of base feature channels for the model body.
        '''
        self.config.models.set_base_channel(base_channel)

    def set_model_bottleneck(self, bottleneck: str) -> None:
        '''
        Set bottleneck architecture type.

        Args:
            bottleneck:
                Registered bottleneck variant identifier.
        '''
        self.config.models.bottleneck = bottleneck

    def set_model_conditioners(self, conditioners: list[str]) -> None:
        '''
        Set list of active feature conditioners.

        Args:
            conditioners:
                Ordered list of conditioner names to apply.
        '''
        self.config.models.conditioners = conditioners

    # ----- bottleneck structure (architecture sub-domain)
    def set_bottleneck_convolution_blocks(self, num_blocks: int | None) -> None:
        '''
        Set convolution block count in the active bottleneck.

        Args:
            num_blocks:
                Number of convolution blocks, or `None` if unused.
        '''
        bottleneck = self.config.models.bottleneck_registry[
            self.config.models.bottleneck
        ]
        bottleneck.num_conv_blocks = num_blocks

    def set_bottleneck_transformer_blocks(self, num_blocks: int | None) -> None:
        '''
        Set transformer block count in the active bottleneck.

        Args:
            num_blocks:
                Number of transformer blocks, or `None` if unused.
        '''
        bottleneck = self.config.models.bottleneck_registry[
            self.config.models.bottleneck
        ]
        bottleneck.num_transformer_blocks = num_blocks

    # ----- transformer parameters (architecture sub-domain)
    def set_transformer_num_heads(self, num_heads: int) -> None:
        '''
        Set transformer attention head count.

        Args:
            num_heads:
                Number of multi-head self-attention heads.
        '''
        bottleneck = self.config.models.bottleneck_registry[
            self.config.models.bottleneck
        ]
        bottleneck.transformer_params.num_heads = num_heads

    def set_transformer_mlp_ratio(self, mlp_ratio: float) -> None:
        '''
        Set transformer MLP expansion ratio.

        Args:
            mlp_ratio:
                Hidden-to-embedding dimension expansion multiplier.
        '''
        bottleneck = self.config.models.bottleneck_registry[
            self.config.models.bottleneck
        ]
        bottleneck.transformer_params.mlp_ratio = mlp_ratio

    def set_transformer_dropout(self, dropout: float) -> None:
        '''
        Set transformer dropout probability.

        Args:
            dropout:
                Dropout probability applied in transformer MLP layers.
        '''
        bottleneck = self.config.models.bottleneck_registry[
            self.config.models.bottleneck
        ]
        bottleneck.transformer_params.dropout = dropout

    def set_transformer_attn_dropout(self, attn_dropout: float) -> None:
        '''
        Set transformer attention dropout probability.

        Args:
            attn_dropout:
                Dropout probability applied to attention weights.
        '''
        bottleneck = self.config.models.bottleneck_registry[
            self.config.models.bottleneck
        ]
        bottleneck.transformer_params.attn_dropout = attn_dropout
