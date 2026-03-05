# =========================================================================== #
#           Copyright (c) His Majesty the King in right of Ontario,           #
#         as represented by the Minister of Natural Resources, 2026.          #
#                                                                             #
#                      © King's Printer for Ontario, 2026.                    #
#                                                                             #
#       Licensed under the Apache License, Version 2.0 (the 'License');       #
#          you may not use this file except in compliance with the            #
#                                  License.                                   #
#                  You may obtain a copy of the License at:                   #
#                                                                             #
#                  http://www.apache.org/licenses/LICENSE-2.0                 #
#                                                                             #
#    Unless required by applicable law or agreed to in writing, software      #
#     distributed under the License is distributed on an 'AS IS' BASIS,       #
#      WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or        #
#                                   implied.                                  #
#       See the License for the specific language governing permissions       #
#                       and limitations under the License.                    #
# =========================================================================== #

'''End-to-end experiment.'''

# standard imports
import os
import typing
# third-party imports
import omegaconf
# local imports
import landseg.controller as controller
import landseg.dataset as dataset
import landseg.training as training
import landseg.utils as utils

def train_end_to_end(config: omegaconf.DictConfig) -> None:
    '''End to end training'''

    # init experiment io folder tree
    exp_dir, log_dir = _init_exp_io(config)

    # create a centralized main logger
    t_stamp = utils.get_timestamp()
    logger = utils.Logger('main', os.path.join(log_dir, f'main_{t_stamp}.log'))

    # data preparation
    data_specs = dataset.load_data(config, logger)

    # build trainer
    trainer = training.build_trainer(data_specs, config, logger)

    # build controller
    runner = controller.build_controller(trainer, config, exp_dir, logger)

    # run via controller
    runner.fit()

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
