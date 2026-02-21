'''Wrapper module to build the multihead model.'''

# third-party imports
import omegaconf
# local imports
import dataset_
import models.multihead.base
import models.multihead.config
import models.multihead.frame

def build_multihead_unet(
        data_specs: dataset_.DataSpecs,
        config: omegaconf.DictConfig,
    ) -> models.multihead.base.BaseModel:
    '''Build the multi-head model from provided dataset.'''

    # general config
    model_config = models.multihead.config.ModelConfig(
        in_ch=data_specs.meta.img_ch_num,
        base_ch=config.channels.base_ch,
        heads_w_counts=data_specs.heads.class_counts,
        enable_logit_adjust=config.logits_adjustment.enable,
        logit_adjust=data_specs.heads.logits_adjust,
        enable_clamp=config.clamp.enable,
        clamp_range=tuple(config.clamp.range),
    )

    # conditioning config
    cond_config = models.multihead.config.CondConfig(
        mode=config.conditioning.mode,
        domain_ids_num=data_specs.domains.ids_max,
        domain_vec_dim=data_specs.domains.vec_dim,
        concat=models.multihead.config.ConcatConfig(
            out_dim=config.conditioning.concat_out_dim,
            use_ids=config.conditioning.concat_use_ids,
            use_vec=config.conditioning.concat_use_vec,
            use_mlp=config.conditioning.concat_use_mlp
        ),
        film=models.multihead.config.FilmConfig(
            embed_dim=config.conditioning.film_embed_dim,
            use_ids=config.conditioning.film_use_ids,
            use_vec=config.conditioning.film_use_vec,
            hidden=config.conditioning.film_hidden
        )
    )

    # return model instance
    return models.multihead.frame.MultiHeadUNet(
        body=config.body,
        config=model_config,
        cond=cond_config,
    )
