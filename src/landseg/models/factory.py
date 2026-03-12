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
import landseg.configs as configs
import landseg.core.ingest_protocols as ingest_protocols
import landseg.models.backbones as backbones
import landseg.models.multihead as multihead

# -------------------------------Public Function-------------------------------
def build_multihead_unet(
    dataspecs: ingest_protocols.DataSpecs,
    config: configs.ModelsCfg,
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
    if not config.use_body in body_registry:
        raise ValueError(f'Invalid base model: {config.use_body}')

    # body configs
    body_config = config.body_registry.get(config.use_body)
    assert body_config is not None # sanity

    # multihead model config
    multihead_config = multihead.ModelConfig(
        body=body_registry[config.use_body],
        in_ch= dataspecs.meta.img_ch,
        base_ch=body_config.base_ch,
        logit_adjust=dataspecs.heads.logits_adjust,
        heads_w_counts=dataspecs.heads.class_counts,
        clamp_range=config.clamp_range,
        conditioning=multihead.CondConfig(
            mode=config.conditioning.mode,
            domain_ids_num=dataspecs.domains.ids_max + 1,
            domain_vec_dim=dataspecs.domains.vec_dim,
            concat=multihead.ConcatConfig(
                out_dim=config.conditioning.concat.out_dim,
                use_ids=config.conditioning.concat.use_ids,
                use_vec=config.conditioning.concat.use_vec,
                use_mlp=config.conditioning.concat.use_mlp
            ),
            film=multihead.FilmConfig(
                embed_dim=config.conditioning.film.embed_dim,
                use_ids=config.conditioning.film.use_ids,
                use_vec=config.conditioning.film.use_vec,
                hidden=config.conditioning.film.hidden
            )
        )
    )

    # return model instance
    return multihead.MultiHeadUNet(
        multihead_config,
        conv_params=body_config.conv_params,
        enable_logit_adjust=config.flags.enable_logit_adjust,
        enable_clamp=config.flags.enable_clamp
    )
