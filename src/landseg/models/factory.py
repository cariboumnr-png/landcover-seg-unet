'''Wrapper module to build the multihead model.'''

# standard imports
import typing
# local imports
import landseg.alias as alias
import landseg.models.multihead as multihead
import landseg.utils as utils

# ---------------------------------Public Type---------------------------------
class ModelDatasetConfig(typing.TypedDict):
    '''Dataset specifications needed to build the model.'''
    img_ch_num: int
    class_counts: dict[str, list[int]]
    logits_adjust: dict[str, list[float]]
    domain_ids_max: int
    domain_vec_dim: int

# -------------------------------Public Function-------------------------------
def build_multihead_unet(
    dataset_config: ModelDatasetConfig,
    model_config: alias.ConfigType
) -> multihead.BaseMultiheadModel:
    '''Build the multi-head model from provided dataset.'''

    # config accessor
    model_cfg = utils.ConfigAccess(model_config)

    # base model config
    base_config = multihead.ModelConfig(
        in_ch=dataset_config['img_ch_num'],
        base_ch=model_cfg.get_option('channels', 'base_ch'),
        heads_w_counts=dataset_config['class_counts'],
        enable_logit_adjust=model_cfg.get_option('logits_adjustment', 'enable'),
        logit_adjust=dataset_config['logits_adjust'],
        enable_clamp=model_cfg.get_option('clamp', 'enable'),
        clamp_range=tuple(model_cfg.get_option('clamp', 'range')),
    )

    # conditioning config
    cond_config = multihead.CondConfig(
        mode=model_cfg.get_option('conditioning', 'mode'),
        domain_ids_num=dataset_config['domain_ids_max'] + 1,
        domain_vec_dim=dataset_config['domain_vec_dim'],
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

    # return model instance
    return multihead.MultiHeadUNet(
        body=model_cfg.get_option('body'),
        config=base_config,
        cond=cond_config,
    )
