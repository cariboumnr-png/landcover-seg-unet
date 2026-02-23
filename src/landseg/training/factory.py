'''Pipeline to build the trainer class.'''

# third-party imports
import omegaconf
# local imports
import landseg.models as models
import landseg.training.callback as callback
import landseg.training.common as common
import landseg.training.dataloading as dataloading
import landseg.training.heads as heads
import landseg.training.loss as loss
import landseg.training.metrics as metrics
import landseg.training.optim as optim
import landseg.training.trainer as trainer
import landseg.utils as utils

# -------------------------------Public Function-------------------------------
def build_trainer(
    data_specs: common.DataSpecsLike,
    config: omegaconf.DictConfig,
    logger: utils.Logger
 ) -> trainer.MultiHeadTrainer:
    '''Builder trainer.'''

    # setup the model
    model = models.build_multihead_unet(
        dataset_config={
            'img_ch_num': data_specs.meta.img_ch_num,
            'class_counts': data_specs.heads.class_counts,
            'logits_adjust': data_specs.heads.logits_adjust,
            'domain_ids_max': data_specs.domains.ids_max,
            'domain_vec_dim': data_specs.domains.vec_dim
        },
        model_config=config.models
    )

    # compile data loaders
    data_loaders = dataloading.get_dataloaders(
        data_specs=data_specs,
        loader_config=config.trainer.loader,
        logger=logger
    )

    # compile training heads basic specifications
    headspecs = heads.build_headspecs(
        data=data_specs,
        config=config.trainer.loss,
    )

    # compile training heads loss compute modules
    headlosses = loss.build_headlosses(
        headspecs=headspecs,
        config=config.trainer.loss.types,
        ignore_index=data_specs.meta.ignore_index,
    )

    # compile training heads metric compute modules
    headmetrics = metrics.build_headmetrics(
        headspecs=headspecs,
        ignore_index=data_specs.meta.ignore_index
    )

    # build optimizer and scheduler
    optimization = optim.build_optimization(
        model=model,
        config=config.trainer.optim
    )

    # generate callback instances
    callbacks = callback.build_callbacks(logger)

    # collect components
    comps = trainer.TrainerComponents(
        model=model,
        dataloaders=data_loaders,
        headspecs=headspecs,
        headlosses=headlosses,
        headmetrics=headmetrics,
        optimization=optimization,
        callbacks=callbacks,
    )

    # parse runtime config
    runtime_cfg = trainer.get_config(config.trainer.runtime)

    # build and return a trainer class
    return trainer.MultiHeadTrainer(comps, runtime_cfg, device='cuda')
