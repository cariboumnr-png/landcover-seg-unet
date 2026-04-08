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

'''
Factory for constructing multihead UNet models.

Provides a factory that assembles:
    - A UNet or UNet++ backbone,
    - Multihead output configuration derived from dataset metadata,
    - Optional domain-conditioning settings (concat / FiLM),
    - Numeric-safety and logit-adjust behaviour.

The primary entry point is `build_multihead_unet`, which returns an
initialized `MultiHeadUNet` instance based on dataset specs and a
user-supplied configuration.
'''

# standard imports
from __future__ import annotations
import typing
# local imports
import landseg.core as core
import landseg.models.multihead as multihead

# --------------------------------private  type--------------------------------
class _BackeboneConfig(typing.Protocol):
    '''Typed container for model backbone configuration.'''
    @property
    def body(self) -> str:...
    @property
    def base_ch(self) -> int:...
    @property
    def conv_params(self) -> dict[str, typing.Any]:...

class _ConditioningConfig(typing.Protocol):
    '''Typed container for model conditioning configuration.'''
    @property
    def mode(self) -> str | None:...
    @property
    def concat(self) -> _ConcatConfig:...
    @property
    def film(self) -> _FilmConfig:...

class _ConcatConfig(typing.Protocol):
    '''Typed container for configuring concatenation adapter.'''
    @property
    def out_dim(self) -> int:...
    @property
    def use_ids(self) -> bool:...
    @property
    def use_vec(self) -> bool:...
    @property
    def use_mlp(self) -> bool:...

class _FilmConfig(typing.Protocol):
    '''Typed container for configuring FiLM conditioner.'''
    @property
    def embed_dim(self) -> int:...
    @property
    def use_ids(self) -> bool:...
    @property
    def use_vec(self) -> bool:...
    @property
    def hidden(self) -> int:...

# -------------------------------Public Function-------------------------------
def build_multihead_unet(
    *,
    dataspecs: core.DataSpecs,
    backbone_config: _BackeboneConfig,
    conditioning_config: _ConditioningConfig,
    enable_logit_adjust: bool,
    enable_clamp: bool,
    clamp_range: tuple[float, float],
) -> multihead.BaseMultiheadModel:
    '''
    Build a configured MultiHeadUNet instance.

    Args:
        body: Backbone type (currently 'unet' or 'unetpp').
        dataspecs: Dataset specs containing image channels, domain, and
            per-head class counts.
        model_config: User configuration describing backbone parameters,
            conditioning options, clamping/logit-adjust flags, and other
            model settings.

    Returns:
        MultiHeadUNet:
            - Selected backbone,
            - Multihead outputs,
            - Input concatenation and/or FiLM conditioning (if enabled),
            - Optional numeric safety and logit-adjust behaviour.

    Raises:
        ValueError: If an unsupported backbone name is provided.
    '''

    # model backbone config
    backbone_config = multihead.BackboneConfig(
        body=backbone_config.body,
        base_ch=backbone_config.base_ch,
        conv_params=backbone_config.conv_params
    )

    # multihead model config
    model_config = multihead.ModelConfig(
        in_ch= dataspecs.meta.img_ch,
        logit_adjust=dataspecs.heads.logits_adjust,
        heads_w_counts=dataspecs.heads.class_counts,
    )

    conditioning=multihead.ConditioningConfig(
            mode=conditioning_config.mode,
            domain_ids_num=dataspecs.domains.ids_max + 1,
            domain_vec_dim=dataspecs.domains.vec_dim,
            concat=multihead.ConcatConfig(
                out_dim=conditioning_config.concat.out_dim,
                use_ids=conditioning_config.concat.use_ids,
                use_vec=conditioning_config.concat.use_vec,
                use_mlp=conditioning_config.concat.use_mlp
            ),
            film=multihead.FilmConfig(
                embed_dim=conditioning_config.film.embed_dim,
                use_ids=conditioning_config.film.use_ids,
                use_vec=conditioning_config.film.use_vec,
                hidden=conditioning_config.film.hidden
            )
        )

    # return model instance
    return multihead.MultiHeadUNet(
        backbone_config=backbone_config,
        model_config=model_config,
        conditioning=conditioning,
        enable_logit_adjust=enable_logit_adjust,
        enable_clamp=enable_clamp,
        clamp_range=clamp_range,
    )
