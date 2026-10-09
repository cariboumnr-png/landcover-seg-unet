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
    - `TrialMutator`: Applies sampled sweep hyperparameters to `RootConfig`.
'''

# standard imports
import typing
# local imports
import landseg.configs as configs
import landseg.configs.schema.study as study_config


class TrialMutator: # pylint: disable=too-many-public-methods
    '''Mutates RootConfig fields during hyperparameter sweep trials.'''

    def __init__(self, config: configs.RootConfig) -> None:
        self.config = config

    @property
    def study(self) -> study_config.StudyConfig:
        '''Expose study configuration search spaces.'''
        return self.config.study

    # ----- data geometry
    def set_data_patch_size(self, patch_size: int) -> None:
        '''Set dataloader patch size.'''
        self.config.session.dataloader.patch_size = patch_size

    def set_data_batch_size(self, batch_size: int) -> None:
        '''Set dataloader batch size.'''
        self.config.session.dataloader.batch_size = batch_size

    # ----- runtime optimization
    def set_optimizer_lr(self, lr: float) -> None:
        '''Set engine optimizer learning rate.'''
        self.config.session.engine_optim.lr = lr

    def set_optimizer_weight_decay(self, weight_decay: float) -> None:
        '''Set engine optimizer weight decay.'''
        self.config.session.engine_optim.weight_decay = weight_decay

    def set_optimizer_type(self, opt_cls: str) -> None:
        '''Set engine optimizer class identifier.'''
        self.config.session.engine_optim.opt_cls = opt_cls

    def set_optimizer_scheduler_type(self, sched_cls: str | None) -> None:
        '''Set engine learning rate scheduler class identifier.'''
        self.config.session.engine_optim.sched_cls = sched_cls

    def set_optimizer_scheduler_args(
        self,
        sched_args: dict[str, typing.Any],
    ) -> None:
        '''Set engine learning rate scheduler arguments.'''
        self.config.session.engine_optim.sched_args = sched_args

    def set_optimizer_grad_clip_norm(
        self,
        grad_clip_norm: float | None,
    ) -> None:
        '''Set engine gradient clipping max norm.'''
        self.config.session.engine_optim.grad_clip_norm = grad_clip_norm

    def set_runtime_use_amp(self, use_amp: bool) -> None:
        '''Set engine mixed precision flag.'''
        self.config.session.engine_exec.use_amp = use_amp

    def set_runtime_logit_adjust_alpha(self, alpha: float) -> None:
        '''Set engine logit adjustment alpha.'''
        self.config.session.engine_exec.logit_adjust_alpha = alpha

    # ----- objective (loss + regularization)
    def set_objective_focal_weight(self, weight: float) -> None:
        '''Set focal loss component weight.'''
        self.config.session.engine_tasks.loss_configs.focal.weight = weight

    def set_objective_dice_weight(self, weight: float) -> None:
        '''Set dice loss component weight.'''
        self.config.session.engine_tasks.loss_configs.dice.weight = weight

    def set_objective_spectral_weight(self, weight: float) -> None:
        '''Set spectral loss component weight.'''
        self.config.session.engine_tasks.loss_configs.spectral.weight = weight

    def set_objective_tv_weight(self, weight: float) -> None:
        '''Set total variation loss component weight.'''
        self.config.session.engine_tasks.loss_configs.tv.weight = weight

    def set_objective_ecological_weight(self, weight: float) -> None:
        '''Set ecological loss component weight.'''
        loss_cfg = self.config.session.engine_tasks.loss_configs
        loss_cfg.ecological.weight = weight

    # ----- multitask
    def set_mtl_consistency_lambda(self, value: float) -> None:
        '''Set MTL consistency regularization lambda.'''
        reg_config = self.config.session.engine_tasks.mtl_reg_configs
        reg_config.consistency_lambda = value

    def set_mtl_consistency_reduction(self, reduction: str) -> None:
        '''Set MTL consistency loss reduction strategy.'''
        reg_config = self.config.session.engine_tasks.mtl_reg_configs
        reg_config.consistency_reduction = reduction

    # ----- architecture
    def set_model_body(self, model_body: str) -> None:
        '''Set UNet backbone model body identifier.'''
        self.config.models.model_body = model_body

    def set_model_base_channel(self, base_channel: int) -> None:
        '''Set base channel count across model architectures.'''
        self.config.models.set_base_channel(base_channel)

    def set_model_bottleneck(self, bottleneck: str) -> None:
        '''Set bottleneck architecture type.'''
        self.config.models.bottleneck = bottleneck

    def set_model_conditioners(self, conditioners: list[str]) -> None:
        '''Set list of active feature conditioners.'''
        self.config.models.conditioners = conditioners

    # ----- bottleneck structure (architecture sub-domain)
    def set_bottleneck_convolution_blocks(self, num_blocks: int | None) -> None:
        '''Set convolution block count in bottleneck.'''
        bottleneck = self.config.models.bottleneck_registry[
            self.config.models.bottleneck
        ]
        bottleneck.num_conv_blocks = num_blocks

    def set_bottleneck_transformer_blocks(self, num_blocks: int | None) -> None:
        '''Set transformer block count in bottleneck.'''
        bottleneck = self.config.models.bottleneck_registry[
            self.config.models.bottleneck
        ]
        bottleneck.num_transformer_blocks = num_blocks

    # ----- transformer parameters (architecture sub-domain)
    def set_transformer_num_heads(self, num_heads: int) -> None:
        '''Set transformer attention head count.'''
        bottleneck = self.config.models.bottleneck_registry[
            self.config.models.bottleneck
        ]
        bottleneck.transformer_params.num_heads = num_heads

    def set_transformer_mlp_ratio(self, mlp_ratio: float) -> None:
        '''Set transformer MLP expansion ratio.'''
        bottleneck = self.config.models.bottleneck_registry[
            self.config.models.bottleneck
        ]
        bottleneck.transformer_params.mlp_ratio = mlp_ratio

    def set_transformer_dropout(self, dropout: float) -> None:
        '''Set transformer dropout probability.'''
        bottleneck = self.config.models.bottleneck_registry[
            self.config.models.bottleneck
        ]
        bottleneck.transformer_params.dropout = dropout

    def set_transformer_attn_dropout(self, attn_dropout: float) -> None:
        '''Set transformer attention dropout probability.'''
        bottleneck = self.config.models.bottleneck_registry[
            self.config.models.bottleneck
        ]
        bottleneck.transformer_params.attn_dropout = attn_dropout
