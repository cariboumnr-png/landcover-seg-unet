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

# -------------------------------Public Function-------------------------------
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
        config=config.experiment,
        logger=logger
    )

def build_trainer(
        trainer_mode: str,
        model: training.common.MultiheadModelLike,
        data_summary: training.common.DataSummaryLike,
        config: omegaconf.DictConfig,
        logger: utils.Logger
    ) -> training.trainer.MultiHeadTrainer:
    '''Builder trainer.'''

    # collect componenets
    trainer_comps = _get_components(
        trainer_mode=trainer_mode,
        model=model,
        data_summary=data_summary,
        config=config,
        logger=logger
    )
    # generate runtime config
    runtime_config = training.trainer.get_config(config.config)

    # build and return a trainer class
    return training.trainer.MultiHeadTrainer(
        components=trainer_comps,
        config=runtime_config,
        device='cuda'
    )

# ------------------------------private  function------------------------------
def _get_components(
        trainer_mode: str,
        model: training.common.MultiheadModelLike,
        data_summary: training.common.DataSummaryLike,
        config: omegaconf.DictConfig,
        logger: utils.Logger
    ) -> training.trainer.TrainerComponents:
    '''Setup the model trainer.'''

    # compile data loaders
    data_loaders = training.dataloading.get_dataloaders(
        mode=trainer_mode,
        data_summary=data_summary,
        loader_config=config.loader,
        logger=logger
    )

    # compile training heads basic specifications
    headspecs = training.heads.build_headspecs(
        data=data_summary,
        config=config.loss.alpha_fn,
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
    optimization = training.optim.build_optimization(
        model=model,
        config=config.optim_config
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
