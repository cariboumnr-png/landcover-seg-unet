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

'''
Training entrypoint.

Builds data specifications from produced artifacts, constructs the
model, and runs the multi-phase training runner.
'''

# standard imports
import dataclasses
import os
# local imports
import landseg.configs as configs
import landseg.factory as factory
import landseg.geopipe as geopipe
import landseg.models as models
import landseg.utils as utils

def train(config: configs.RootConfig):
    '''
    Run a full training job.

    Creates an experiment directory, builds `DataSpecs` from the prepared
    artifacts and schema, instantiates the model, and executes the runner.

    Args:
        config: RootConfig with model, trainer, and runner settings.
    '''

    # init experiment io folder tree
    exp_dir, log_dir = _init_experiment_folder(config)

    # create a centralized main logger
    t_stamp = utils.get_timestamp()
    logger = utils.Logger('main', os.path.join(log_dir, f'main_{t_stamp}.log'))

    # collect artifacts and build dataspsec
    dataspecs = geopipe.build_dataspec(
        f'{config.foundation.output_dpath}/data_blocks/model_dev/metadata.json',
        f'{config.foundation.output_dpath}/domain_knowledge/ecodistrict.json',
        f'{config.foundation.output_dpath}/domain_knowledge/geology.json',
        f'{config.transform.output_dpath}/schema.json',
        print_out=True
    )

    # setup the model
    model = models.build_multihead_unet(dataspecs, config.models)

    # build controller
    runner = factory.build_runner(exp_dir, dataspecs, model, config, logger)

    # run via controller
    runner.fit()

def _init_experiment_folder(config: configs.RootConfig) -> tuple[str, str]:
    '''Initialize experiment directories.'''

    # get from config
    exp_root = config.exp_root

    # ensure output folders exist (e.g, for fresh experiment)
    # top-level
    results = os.path.join(exp_root, 'results')
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
    config_dict = dataclasses.asdict(config)
    utils.write_json(os.path.join(exp_dir, 'config.json'), config_dict)
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
