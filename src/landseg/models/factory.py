'''Wrapper module to build the multihead model.'''

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
    '''Build the multi-head model from provided dataset.'''

    # model body registry
    body_registry = {
        'unet': backbones.UNet,
        'unetpp': backbones.UNetPP
    }
    assert body in body_registry, f'Invalid base model: {body}'

    # config accessors
    model_cfg = utils.ConfigAccess(model_config)

    # parse from config and dataspecs
    model_base_channel = model_cfg.get_option('body', body, 'base_ch')
    model_conditioning = _conditioning_config(model_cfg, dataspecs)
    model_clamp_range = tuple(model_cfg.get_option('clamp_range'))

    # multihead model config
    multihead_config = multihead.ModelConfig(
        body_registry[body],
        dataspecs.meta.img_ch_num,
        model_base_channel,
        dataspecs.heads.logits_adjust,
        dataspecs.heads.class_counts,
        model_conditioning,
        model_clamp_range,
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
    '''Helper to configure model conditioning.'''

    return multihead.CondConfig(
        model_cfg.get_option('conditioning', 'mode'),
        dataspecs.domains.ids_max + 1,
        dataspecs.domains.vec_dim,
        multihead.ConcatConfig(
            model_cfg.get_option('conditioning', 'concat_out_dim'),
            model_cfg.get_option('conditioning', 'concat_use_ids'),
            model_cfg.get_option('conditioning', 'concat_use_vec'),
            model_cfg.get_option('conditioning', 'concat_use_mlp')
        ),
        multihead.FilmConfig(
            model_cfg.get_option('conditioning', 'film_embed_dim'),
            model_cfg.get_option('conditioning', 'film_use_ids'),
            model_cfg.get_option('conditioning', 'film_use_vec'),
            model_cfg.get_option('conditioning', 'film_hidden')
        )
    )
