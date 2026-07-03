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
# pylint: disable=too-many-instance-attributes
# pylint: disable=too-many-public-methods

'''
This module mirrors the Hydra/YAML config tree using Python dataclasses,
suitable for OmegaConf structured configs.
'''

# standard imports
from __future__ import annotations
import dataclasses
import typing
# local imports
import landseg.configs.schema.sections as sec

# alias
field = dataclasses.field

# ------------------------------EXECUTION CONFIGS------------------------------
@dataclasses.dataclass
class _ExecutionContext:
    '''Mutable execution context.'''
    verbosity: str = 'full' # 'full', 'logging_only', 'silent'
    exp_root: str = './experiment' # root directory for this experiment run
    user_cfg: str | None = None # external user configs
    dev_cfg: str | None = None # developer-only override config
    cli_mode: bool = False # indicates whether the execution was initiated through the CLI resolver

# --------------------------------ROOT  CONFIGS--------------------------------
@dataclasses.dataclass
class RootConfig:
    '''Root structured config for landseg.'''

    # execution configs
    execution: _ExecutionContext = field(default_factory=_ExecutionContext)
    # raw input data and configs
    foundation: sec.DataFoundation = field(default_factory=sec.DataFoundation)
    # data preparation
    transform: sec.DataTransform = field(default_factory=sec.DataTransform)
    # data specfication
    dataspecs: sec.DataSpecs = field(default_factory=sec.DataSpecs)
    # model settings
    models: sec.ModelsConfig = field(default_factory=sec.ModelsConfig)
    # session settings
    session: sec.SessionConfig = field(default_factory=sec.SessionConfig)
    # study settings
    study: sec.StudyConfig = field(default_factory=sec.StudyConfig)
    # pipeline specific CLI flags
    pipeline: sec.PipelineConfig = field(default_factory=sec.PipelineConfig)

    @property
    def as_dict(self) -> dict[str, typing.Any]:
        return dataclasses.asdict(self)

    def validate_all(self) -> None:
        # To be completed for all sections
        self.foundation.validate()
        self.transform.validate()
        self.models.validate()
        self.session.validate()

    # hyperparameter setters (for sweeping)
    # ----- data geometry
    def set_data_patch_size(self, patch_size: int) -> None:
        self.session.data_loader.patch_size = patch_size

    def set_data_batch_size(self, batch_size: int) -> None:
        self.session.data_loader.batch_size = batch_size

    # ----- runtime optimization
    def set_optimizer_lr(self, lr: float) -> None:
        self.session.engine_optim.lr = lr

    def set_optimizer_weight_decay(self, weight_decay: float) -> None:
        self.session.engine_optim.weight_decay = weight_decay

    def set_optimizer_type(self, opt_cls: str) -> None:
        self.session.engine_optim.opt_cls = opt_cls

    def set_optimizer_scheduler_type(self, sched_cls: str | None) -> None:
        self.session.engine_optim.sched_cls = sched_cls

    def set_optimizer_scheduler_args(self, sched_args: dict[str, typing.Any]) -> None:
        self.session.engine_optim.sched_args = sched_args

    def set_optimizer_grad_clip_norm(self, grad_clip_norm: float | None) -> None:
        self.session.engine_optim.grad_clip_norm = grad_clip_norm

    def set_runtime_use_amp(self, use_amp: bool) -> None:
        self.session.engine_exec.use_amp = use_amp

    def set_runtime_logit_adjust_alpha(self, alpha: float) -> None:
        self.session.engine_exec.logit_adjust_alpha = alpha

    # ----- objective (loss + regularization)
    def set_objective_focal_weight(self, weight: float) -> None:
        self.session.engine_tasks.loss_configs.focal.weight = weight

    def set_objective_dice_weight(self, weight: float) -> None:
        self.session.engine_tasks.loss_configs.dice.weight = weight

    def set_objective_spectral_weight(self, weight: float) -> None:
        self.session.engine_tasks.loss_configs.spectral.weight = weight

    def set_objective_tv_weight(self, weight: float) -> None:
        self.session.engine_tasks.loss_configs.tv.weight = weight

    # ----- multitask
    def set_mtl_consistency_lambda(self, value: float) -> None:
        reg_config = self.session.engine_tasks.mtl_reg_configs
        reg_config.consistency_lambda = value

    def set_mtl_consistency_reduction(self, reduction: str) -> None:
        reg_config = self.session.engine_tasks.mtl_reg_configs
        reg_config.consistency_reduction = reduction

    # ----- architecture
    def set_model_body(self, model_body: str) -> None:
        self.models.model_body = model_body

    def set_model_base_channel(self, base_channel: int) -> None:
        self.models.set_base_channel(base_channel)

    def set_model_bottleneck(self, bottleneck: str) -> None:
        self.models.bottleneck = bottleneck

    def set_model_conditioners(self, conditioners: list[str]) -> None:
        self.models.conditioners = conditioners

    # ----- bottleneck structure (architecture sub-domain)
    def set_bottleneck_convolution_blocks(self, num_blocks: int | None) -> None:
        bottleneck = self.models.bottleneck_registry[self.models.bottleneck]
        bottleneck.num_conv_blocks = num_blocks

    def set_bottleneck_transformer_blocks(self, num_blocks: int | None) -> None:
        bottleneck = self.models.bottleneck_registry[self.models.bottleneck]
        bottleneck.num_transformer_blocks = num_blocks

    # ----- transformer parameters (architecture sub-domain)
    def set_transformer_num_heads(self, num_heads: int) -> None:
        bottleneck = self.models.bottleneck_registry[self.models.bottleneck]
        bottleneck.transformer_params.num_heads = num_heads

    def set_transformer_mlp_ratio(self, mlp_ratio: float) -> None:
        bottleneck = self.models.bottleneck_registry[self.models.bottleneck]
        bottleneck.transformer_params.mlp_ratio = mlp_ratio

    def set_transformer_dropout(self, dropout: float) -> None:
        bottleneck = self.models.bottleneck_registry[self.models.bottleneck]
        bottleneck.transformer_params.dropout = dropout

    def set_transformer_attn_dropout(self, attn_dropout: float) -> None:
        bottleneck = self.models.bottleneck_registry[self.models.bottleneck]
        bottleneck.transformer_params.attn_dropout = attn_dropout
