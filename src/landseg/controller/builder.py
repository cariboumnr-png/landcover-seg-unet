'''Build a controller.'''

# third-party imports
import omegaconf
# local imports
import landseg.controller as controller
import landseg.training as training
import landseg.utils as utils

def build_controller(
    engine: training.MultiHeadTrainer,
    config: omegaconf.DictConfig,
    logger: utils.Logger
) -> controller.Controller:
    '''Setup training controller.'''

    # get phases
    phases = controller.generate_phases(config.curriculum)

    # return a controller (as the main runner)
    return controller.Controller(
        engine=engine,
        phases=phases,
        config=config.curriculum.experiment,
        logger=logger
    )
