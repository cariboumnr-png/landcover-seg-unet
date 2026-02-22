'''Pipeline to build the trainer class.'''

# third-party imports
import omegaconf
# local imports
import models
import training.callback
import training.common
import training.controller
import training.dataloading
import training.heads
import training.loss
import training.metrics
import training.optim
import training.trainer
import utils

# -------------------------------Public Function-------------------------------
def build_runner(
    data_specs: training.common.DataSpecsLike,
    config: omegaconf.DictConfig,
    logger: utils.Logger
) -> training.controller.Controller:
    '''Setup training controller.'''

    # build trainer
    trainer = _build_trainer(data_specs, config, logger)

    # get phases
    phases = training.controller.generate_phases(config.curriculum)

    # return a controller (as the main runner)
    return training.controller.Controller(
        trainer=trainer,
        phases=phases,
        config=config.curriculum.experiment,
        logger=logger
    )

def _build_trainer(
    data_specs: training.common.DataSpecsLike,
    config: omegaconf.DictConfig,
    logger: utils.Logger
) -> training.trainer.MultiHeadTrainer:
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
    data_loaders = training.dataloading.get_dataloaders(
        data_specs=data_specs,
        loader_config=config.trainer.loader,
        logger=logger
    )

    # compile training heads basic specifications
    headspecs = training.heads.build_headspecs(
        data=data_specs,
        config=config.trainer.loss,
    )

    # compile training heads loss compute modules
    headlosses = training.loss.build_headlosses(
        headspecs=headspecs,
        config=config.trainer.loss.types,
        ignore_index=data_specs.meta.ignore_index,
    )

    # compile training heads metric compute modules
    headmetrics = training.metrics.build_headmetrics(
        headspecs=headspecs,
        ignore_index=data_specs.meta.ignore_index
    )

    # build optimizer and scheduler
    optimization = training.optim.build_optimization(
        model=model,
        config=config.trainer.optim
    )

    # generate callback instances
    callbacks = training.callback.build_callbacks(logger)

    # collect components
    comps = training.trainer.TrainerComponents(
        model=model,
        dataloaders=data_loaders,
        headspecs=headspecs,
        headlosses=headlosses,
        headmetrics=headmetrics,
        optimization=optimization,
        callbacks=callbacks,
    )

    # parse runtime config
    runtime_cfg = training.trainer.get_config(config.trainer.runtime)

    # build and return a trainer class
    return training.trainer.MultiHeadTrainer(comps, runtime_cfg, device='cuda')
