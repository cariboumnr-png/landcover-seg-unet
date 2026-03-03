'''End-to-end experiment.'''

# third-party imports
import omegaconf
# local imports
import landseg.controller as controller
import landseg.dataset as dataset
import landseg.training as training
import landseg.utils as utils

def train_end_to_end(
    exp_dir: str,
    config: omegaconf.DictConfig,
    logger: utils.Logger
):
    '''End to end training'''
    # data preparation
    data_specs = dataset.load_data(config, logger)

    # build trainer
    trainer = training.build_trainer(data_specs, config, logger)

    # build controller
    runner = controller.build_controller(trainer, config, exp_dir, logger)

    # run via controller
    runner.fit()
