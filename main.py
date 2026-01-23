# pylint: disable=no-value-for-parameter
'''Main entry point.'''

# third-party imports
import hydra
import omegaconf
# local imports
import src.dataset
import src.models
import src.training

# add resolvers to omegaconf for conveniences
#  - convert decimals to percentages
omegaconf.OmegaConf.register_new_resolver('c', lambda x: int(x * 100))

# main process
@hydra.main(config_path='./configs', config_name='main', version_base='1.3')
def main(config: omegaconf.DictConfig):
    '''doc'''

    # data preparation
    data_summary = src.dataset.prepare_data(config.dataset)

    # setup multihead model
    model = src.models.multihead_unet(data_summary, config.models)

    # setup trainer
    trainer = src.training.build_trainer(model, data_summary, config.trainer)

    # setup curriculum
    controller = src.training.build_controller(trainer, config.curriculum)

    # start
    controller.fit()

if __name__ == '__main__':
    main() # supposed to run without arguments
