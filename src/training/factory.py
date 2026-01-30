'''Pipeline to build the trainer class.'''

# third-party imports
import omegaconf
# local imports
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

def get_components(
        model: training.common.MultiheadModelLike,
        data_summary: training.common.DataSummaryFull,
        config: omegaconf.DictConfig,
        logger: utils.Logger
    ) -> training.trainer.TrainerComponents:
    '''Setup the model trainer.'''

    # compile data loaders
    data_loaders = training.dataloading.get_dataloaders(
        data_summary=data_summary,
        loader_config=config.loader,
        logger=logger
    )

    # compile training heads basic specifications
    headspecs = training.heads.build_headspecs(
        data=data_summary,
        alpha_fn=config.loss.alpha_fn,
    )

    # compile training heads loss compute modules
    headlosses = training.loss.build_headlosses(
        headspecs=headspecs,
        config=config.loss.types,
        ignore_index=data_summary.meta.ignore_index,
    )

    # compile training heads metric compute modules
    headmetrics = training.metrics.build_headmetrics(
        headspecs=headspecs,
        ignore_index=data_summary.meta.ignore_index
    )

    # build optimizer and scheduler
    optim_config = training.optim.build_optim_config(
        opt_cls=config.optim.opt_cls,
        lr=config.optim.lr,
        weight_decay=config.optim.weight_decay,
        sched_cls=config.optim.sched_cls,
        sched_args=config.optim.sched_args
    )
    optimization = training.optim.build_optimization(
        model=model,
        config=optim_config
    )

    # generate callback instances
    callbacks = training.callback.build_callbacks(logger)

    # collect components and return
    components = training.trainer.TrainerComponents(
        model=model,
        dataloaders=data_loaders,
        headspecs=headspecs,
        headlosses=headlosses,
        headmetrics=headmetrics,
        optimization=optimization,
        callbacks=callbacks,
    )
    return components

def get_config(config: omegaconf.DictConfig) -> training.trainer.RuntimeConfig:
    '''Generate trainer runtime config from hydra.'''

    running_config = training.trainer.RuntimeConfig()
    # assign domain config
    running_config.data.dom_ids_name=config.data.domain_ids_name
    running_config.data.dom_vec_name=config.data.domain_vec_name
    # assign schedule config
    running_config.schedule.max_epoch=config.schedule.max_epoch
    running_config.schedule.max_step=config.schedule.max_step
    running_config.schedule.logging_interval=config.schedule.log_every
    running_config.schedule.eval_interval=config.schedule.val_every
    running_config.schedule.checkpoint_interval=config.schedule.ckpt_every
    running_config.schedule.patience_epochs=config.schedule.patience
    running_config.schedule.min_delta=config.schedule.min_delta
    # assign monitoring config
    running_config.monitor.enabled=('iou',)
    running_config.monitor.metric=config.monitor.metric_name
    running_config.monitor.head=config.monitor.track_head_name
    running_config.monitor.mode=config.monitor.track_mode
    # assign precision config
    running_config.precision.use_amp=config.precision.use_amp
    # assign optimization config
    running_config.optim.grad_clip_norm=config.optimization.grad_clip_norm
    # return complete runnning config
    return running_config

def build_trainer(
        model: training.common.MultiheadModelLike,
        data_summary: training.common.DataSummaryFull,
        config: omegaconf.DictConfig,
        logger: utils.Logger
    ) -> training.trainer.MultiHeadTrainer:
    '''Builder trainer.'''

    # collect componenets
    trainer_comps = get_components(
        model=model,
        data_summary=data_summary,
        config=config,
        logger=logger
    )
    # generate runtime config
    runtime_config = get_config(config.config)

    # build and return a trainer class
    return training.trainer.MultiHeadTrainer(
        components=trainer_comps,
        config=runtime_config,
        device='cuda'
    )

def build_controller(
        trainer: training.trainer.MultiHeadTrainer,
        config: omegaconf.DictConfig,
        logger: utils.Logger
    ) -> training.controller.Controller:
    '''Setup training controller.'''

    # get phases
    phases = training.controller.generate_phases(config)

    # return a controller class
    return training.controller.Controller(
        trainer=trainer,
        phases=phases,
        ckpt_dpath=config.ckpt_dpath,
        logger=logger
    )
