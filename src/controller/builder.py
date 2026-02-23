'''Build a controller.'''

# third-party imports
import omegaconf
# local imports
import controller
import training
import utils

def build_controller(
    trainer: training.MultiHeadTrainer,
    config: omegaconf.DictConfig,
    logger: utils.Logger
) -> controller.Controller:
    '''Setup training controller.'''

    # get phases
    phases = controller.generate_phases(config.curriculum)

    # return a controller (as the main runner)
    return controller.Controller(
        trainer=trainer,
        phases=phases,
        config=config.curriculum.experiment,
        logger=logger
    )
