'''Wrapper module to build the multihead model.'''

# third-party imports
import omegaconf
# local imports
import dataset.summary
import models.multihead.base
import models.multihead.config
import models.multihead.frame

def build_multihead_unet(
        data_summary: dataset.summary.DataSummary,
        config: omegaconf.DictConfig,
    ) -> models.multihead.base.BaseModel:
    '''Build the multi-head model from provided dataset.'''

    # general config
    model_config = models.multihead.config.ModelConfig(
        base_ch=config.channels.base_ch,
        enable_logit_adjust=config.logits_adjustment.enable,
        enable_clamp=config.clamp.enable,
        clamp_range=tuple(config.clamp.range),
        in_ch=data_summary.meta.img_ch_num,
        heads_w_counts=data_summary.heads.class_counts,
        logit_adjust=data_summary.heads.logits_adjust,
    )

    # conditioning config
    cond_config = models.multihead.config.CondConfig(
        mode=config.conditioning.mode,
        domain_ids_num=config.conditioning.domain_ids_num,
        domain_vec_dim=config.conditioning.domain_vec_dim,
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
