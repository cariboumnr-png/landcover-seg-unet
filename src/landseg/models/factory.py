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
    **kwargs
) -> multihead.BaseMultiheadModel:
    '''
    Construct a configured MultiHeadUNet from explicit inputs.

    This factory assembles a complete multi-head UNet model using:
    - dataset-derived runtime specifications (`DataSpecs`), and
    - explicitly supplied configuration objects describing backbone
      structure and domain conditioning behavior.

    The factory is intentionally decoupled from any global or external
    configuration system (e.g., Hydra). Configuration inputs are treated
    as structural contracts (via Protocols) and may originate from
    Hydra-backed dataclasses, plain dataclasses, or other compatible
    objects defined at the application / CLI layer.

    Responsibilities of this factory:
        - Translate dataset metadata into model-level configuration
          (input channels, head definitions, logit-adjust priors).
        - Normalize backbone and conditioning configuration into the
          internal `multihead.*Config` dataclasses owned by the models
          module.
        - Instantiate and return a fully initialized `MultiHeadUNet`.

    Args:
        dataspecs:
            Dataset specifications carrying runtime information derived
            from the data pipeline, including:
                - image channel count,
                - per-head class counts,
                - optional logit-adjust priors,
                - domain cardinalities and vector dimensions.
        backbone_config:
            Backbone configuration describing:
                - backbone body name (e.g., 'unet', 'unetpp'),
                - base channel width,
                - convolutional block parameters forwarded to the
                  backbone constructor.
        conditioning_config:
            Domain conditioning configuration describing how categorical
            domain IDs and/or continuous domain vectors are routed to:
                - input-level concatenation, and/or
                - bottleneck-level FiLM conditioning.
        **kwargs:
            Optional runtime overrides forwarded to `MultiHeadUNet`,
            such as:
                - `enable_logit_adjust`,
                - `enable_clamp`,
                - `clamp_range`.

    Returns:
        BaseMultiheadModel:
            A fully configured `MultiHeadUNet` instance composed of:
                - the selected UNet backbone,
                - multihead output blocks,
                - optional domain conditioning (concat / FiLM),
                - numeric safety and logit-adjust mechanisms.

    Raises:
        ValueError:
            If an unsupported backbone identifier is provided.

    Notes:
        - All arguments are keyword-only to make configuration boundaries
          explicit and order-independent.
        - This factory assumes configuration objects are already
          validated by the application layer.
        - No Hydra or experiment-level configuration is imported or
          accessed within this module by design.
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
        **kwargs
    )
