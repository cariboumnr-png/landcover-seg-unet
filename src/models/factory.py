'''Wrapper module to build the multihead model.'''

# standard imports
import typing
# local imports
import alias
import models.multihead.base
import models.multihead.config
import models.multihead.frame
import utils

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
) -> models.multihead.base.BaseModel:
    '''Build the multi-head model from provided dataset.'''

    # config accessor
    model_cfg = utils.ConfigAccess(model_config)

    # base model config
    base_config = models.multihead.config.ModelConfig(
        in_ch=dataset_config['img_ch_num'],
        base_ch=model_cfg.get_option('channels', 'base_ch'),
        heads_w_counts=dataset_config['class_counts'],
        enable_logit_adjust=model_cfg.get_option('logits_adjustment', 'enable'),
        logit_adjust=dataset_config['logits_adjust'],
        enable_clamp=model_cfg.get_option('clamp', 'enable'),
        clamp_range=tuple(model_cfg.get_option('clamp', 'range')),
    )

    # conditioning config
    cond_config = models.multihead.config.CondConfig(
        mode=model_cfg.get_option('conditioning', 'mode'),
        domain_ids_num=dataset_config['domain_ids_max'],
        domain_vec_dim=dataset_config['domain_vec_dim'],
        concat=models.multihead.config.ConcatConfig(
            out_dim=model_cfg.get_option('conditioning', 'concat_out_dim'),
            use_ids=model_cfg.get_option('conditioning', 'concat_use_ids'),
            use_vec=model_cfg.get_option('conditioning', 'concat_use_vec'),
            use_mlp=model_cfg.get_option('conditioning', 'concat_use_mlp')
        ),
        film=models.multihead.config.FilmConfig(
            embed_dim=model_cfg.get_option('conditioning', 'film_embed_dim'),
            use_ids=model_cfg.get_option('conditioning', 'film_use_ids'),
            use_vec=model_cfg.get_option('conditioning', 'film_use_vec'),
            hidden=model_cfg.get_option('conditioning', 'film_hidden')
        )
    )

    # return model instance
    return models.multihead.frame.MultiHeadUNet(
        body=model_cfg.get_option('body'),
        config=base_config,
        cond=cond_config,
    )
