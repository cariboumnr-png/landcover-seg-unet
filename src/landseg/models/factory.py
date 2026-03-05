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

# local imports
import landseg.alias as alias
import landseg.core as core
import landseg.models.backbones as backbones
import landseg.models.multihead as multihead
import landseg.utils as utils

# -------------------------------Public Function-------------------------------
def build_multihead_unet(
    body: str,
    dataspecs: core.DataSpecsLike,
    model_config: alias.ConfigType,
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

    # model body registry
    body_registry = {
        'unet': backbones.UNet,
        'unetpp': backbones.UNetPP
    }
    if not body in body_registry:
        raise ValueError(f'Invalid base model: {body}')

    # config accessors
    model_cfg = utils.ConfigAccess(model_config)

    # multihead model config
    multihead_config = multihead.ModelConfig(
        body=body_registry[body],
        in_ch= dataspecs.meta.img_ch_num,
        base_ch=model_cfg.get_option('body', body, 'base_ch'),
        logit_adjust=dataspecs.heads.logits_adjust,
        heads_w_counts=dataspecs.heads.class_counts,
        conditioning=_conditioning_config(model_cfg, dataspecs),
        clamp_range=tuple(model_cfg.get_option('clamp_range')),
    )

    # keyword arguments
    enable_logit_adjust = model_cfg.get_option('flags', 'enable_logit_adjust')
    enable_clamp = model_cfg.get_option('flags', 'enable_clamp')
    model_conv_params = model_cfg.get_option('body', body, 'conv_params')

    # return model instance
    return multihead.MultiHeadUNet(
        multihead_config,
        conv_params=model_conv_params,
        enable_logit_adjust=enable_logit_adjust,
        enable_clamp=enable_clamp
    )

def _conditioning_config(
    model_cfg: utils.ConfigAccess,
    dataspecs: core.DataSpecsLike
) -> multihead.CondConfig:
    '''Configure model conditioning components.'''

    return multihead.CondConfig(
        mode=model_cfg.get_option('conditioning', 'mode'),
        domain_ids_num=dataspecs.domains.ids_max + 1,
        domain_vec_dim=dataspecs.domains.vec_dim,
        concat=multihead.ConcatConfig(
            out_dim=model_cfg.get_option('conditioning', 'concat_out_dim'),
            use_ids=model_cfg.get_option('conditioning', 'concat_use_ids'),
            use_vec=model_cfg.get_option('conditioning', 'concat_use_vec'),
            use_mlp=model_cfg.get_option('conditioning', 'concat_use_mlp')
        ),
        film=multihead.FilmConfig(
            embed_dim=model_cfg.get_option('conditioning', 'film_embed_dim'),
            use_ids=model_cfg.get_option('conditioning', 'film_use_ids'),
            use_vec=model_cfg.get_option('conditioning', 'film_use_vec'),
            hidden=model_cfg.get_option('conditioning', 'film_hidden')
        )
    )
