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
import landseg.cli as cli
import landseg.utils as utils

CMD = {
    'end-to-end': cli.train_end_to_end,
    'overfit-test': ''
}

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
                raise TypeError('./settings.yaml must have a mapping')
            # allow new phases to be added
            with omegaconf.open_dict(config.experiment.phases):
                merged = omegaconf.OmegaConf.merge(config, user_cfg) # right wins
                config = typing.cast(omegaconf.DictConfig, merged)
    omegaconf.OmegaConf.resolve(config)

    # init experiment io folder tree
    exp_dir, log_dir = _init_exp_io(config)

    # create a centralized main logger
    t_stamp = utils.get_timestamp()
    logger = utils.Logger('main', os.path.join(log_dir, f'main_{t_stamp}.log'))

    # run specified mode with exceptions handling
    mode = config['run_mmode']
    try:
        CMD[mode](exp_dir, config, logger)
    # manual keyboard interruption
    except KeyboardInterrupt:
        logger.log('INFO', '\nExperiment manually interrupted, exiting...')
        sys.exit(130)
    # capture others and log
    except Exception: # pylint: disable=broad-exception-caught
        logger.log('CRITICAL', 'Unhandled exception occurred', exc_info=True)
        sys.exit(1)

def _init_exp_io(config: omegaconf.DictConfig) -> tuple[str, str]:
    '''Initialize experiment I/O folder tree and lazily check inputs.'''

    # get from config
    exp_root = config['exp_root']
    dataset_name = config['dataset']['name']

    # lazy check if mandatory inputs are present
    # check input fit rasters
    input_fit_dir = os.path.join(exp_root, 'input', dataset_name, 'fit')
    if not _check_file_types_in_dir(('tif', 'tiff'), input_fit_dir):
        raise ValueError(f'No rasters (.tif) found at {input_fit_dir}')
    # check input configs
    input_cfg_dir = os.path.join(exp_root, 'input', dataset_name, 'configs')
    if not _check_file_types_in_dir(('json',), input_cfg_dir):
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
        exp_dir = os.path.join(results, f'exp_{i:04d}')
        try:
            os.makedirs(exp_dir)
            break
        except FileExistsError:
            i += 1
    # save running config per experiment
    _config = omegaconf.OmegaConf.to_container(config, resolve=True)
    _config = typing.cast(dict, _config)
    utils.write_json(os.path.join(exp_dir, 'config.json'), _config)
    # experiment components
    logs_dir = os.path.join(exp_dir, 'logs')
    ckpt_dir = os.path.join(exp_dir, 'checkpoints')
    prev_dir = os.path.join(exp_dir, 'previews')
    plot_dir = os.path.join(exp_dir, 'plots')
    os.makedirs(logs_dir, exist_ok=True)
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(prev_dir, exist_ok=True)
    os.makedirs(plot_dir, exist_ok=True)

    # return experiment dir and log dir
    return exp_dir, logs_dir

def _check_file_types_in_dir(suffixes: tuple[str, ...], dirpath: str) -> bool:
    '''Check if files of select suffixes are present in directory.'''

    if not os.path.exists(dirpath):
        return False
    if not any(
        any(name.endswith(s) for s in suffixes) for name in os.listdir(dirpath)
        if os.path.isfile(os.path.join(dirpath, name))
    ):
        return False
    return True


if __name__ == '__main__':
    main()
