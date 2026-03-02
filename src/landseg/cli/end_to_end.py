# pylint: disable=no-value-for-parameter
'''End-to-end experiment.'''

# standard imports
import os
import sys
import typing
# third-party imports
import hydra
import hydra.utils
import omegaconf
# local imports
import landseg.controller as controller
import landseg.dataset as dataset
import landseg.training as training
import landseg.utils as utils

# main process
@hydra.main('pkg://landseg/configs', 'config', version_base='1.3')
def main(config: omegaconf.DictConfig) -> None:
    '''End-to-end experiment.'''

    # safer CWD fetching
    original_cwd = hydra.utils.get_original_cwd()

    # user settings at root
    candidates = [os.path.join(original_cwd, 'settings.yaml')]
    # optional dev settings (untracked, supplied via CLI argument)
    aux = config.get('dev_settings_path')
    if aux:
        aux = aux if os.path.isabs(aux) else os.path.join(original_cwd, aux)
    candidates.append(aux)

    # merging overrides with default config tree and resolve
    for p in candidates:
        if os.path.exists(p):
            user_cfg = omegaconf.OmegaConf.load(p)
            if not isinstance(user_cfg, omegaconf.DictConfig):
                raise TypeError('settings.yaml must have a mapping at the root')
            # allow new phases to be added
            with omegaconf.open_dict(config.curriculum.phases):
                merged = omegaconf.OmegaConf.merge(config, user_cfg) # right wins
                config = typing.cast(omegaconf.DictConfig, merged)
    omegaconf.OmegaConf.resolve(config)

    # init io folder tree and create a centralized logger file
    logger = init_exp_io(config)

    # run exceptions handling
    try:
        # data preparation
        data_specs = dataset.load_data(config, logger)

        # build trainer
        trainer = training.build_trainer(data_specs, config, logger)

        # build controller
        runner = controller.build_controller(trainer, config, logger)

        # run via controller
        runner.fit()
    # manual keyboard interruption
    except KeyboardInterrupt:
        logger.log('INFO', '\nExperiment manually interrupted, exiting...')
        sys.exit(130)
    # capture others and log
    except Exception: # pylint: disable=broad-exception-caught
        logger.log('CRITICAL', 'Unhandled exception occurred', exc_info=True)
        sys.exit(1)

def init_exp_io(config: omegaconf.DictConfig) -> utils.Logger:
    '''Initialize experiment I/O folder tree and lazily check inputs.'''

    # get from config
    exp_root = config['exp_root']
    dataset_name = config['dataset']['name']

    # lazy check if mandatory inputs are present
    # check input fit rasters
    input_fit_dir = os.path.join(exp_root, 'input', dataset_name, 'fit')
    if not os.path.exists(input_fit_dir):
        raise ValueError(f'Input fit raster root not found: {input_fit_dir}')
    if not any(
        name.endswith('.tif') or name.endswith('.tiff')
        for name in os.listdir(input_fit_dir)
        if os.path.isfile(os.path.join(input_fit_dir, name))
    ):
        raise ValueError(f'No rasters (.tif) found at {input_fit_dir}')
    # check input configs
    input_cfg_dir = os.path.join(exp_root, 'input', dataset_name, 'configs')
    if not os.path.exists(input_cfg_dir):
        raise ValueError(f'Input configs root not found: {input_cfg_dir}')
    if not any(
        name.endswith('.json')
        for name in os.listdir(input_cfg_dir)
        if os.path.isfile(os.path.join(input_cfg_dir, name))
    ):
        raise ValueError(f'No data configs (.json) found at {input_cfg_dir}')

    # ensure output folders exist (e.g, for fresh experiment)
    # top-level
    artifacts = os.path.join(exp_root, 'artifacts')
    results = os.path.join(exp_root, 'results')
    os.makedirs(artifacts, exist_ok=True)
    os.makedirs(results, exist_ok=True)
    # experiment root - natural counter from 0001 to 9999
    i = 1
    while True:
        experiment = os.path.join(results, f'exp_{i:04d}')
        try:
            os.makedirs(experiment)
            break
        except FileExistsError:
            i += 1
    # save running config per experiment
    utils.write_json(os.path.join(experiment, 'config.json'), config)
    # experiment components
    logs = os.path.join(experiment, 'logs')
    ckpt = os.path.join(experiment, 'checkpoints')
    prev = os.path.join(experiment, 'previews')
    plot = os.path.join(experiment, 'plots')
    os.makedirs(logs, exist_ok=True)
    os.makedirs(ckpt, exist_ok=True)
    os.makedirs(prev, exist_ok=True)
    os.makedirs(plot, exist_ok=True)

    # create a centralized main logger and return
    timestamp = utils.get_timestamp()
    log_file = os.path.join(logs, f'main_{timestamp}.log')
    return utils.Logger('main', log_file)

if __name__ == '__main__':
    main()
