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
# third-party imports
import torch
# local imports
import landseg.artifacts as artifacts
import landseg.configs as configs
import landseg.geopipe as geopipe
import landseg.models as models
import landseg.session as session
import landseg.utils as utils

def train(config: configs.RootConfig):
    '''
    Run a full training job.

    Creates an run directory, builds `DataSpecs` from the prepared
    artifacts and schema, instantiates the model, and executes the runner.

    Args:
        config: RootConfig with model, trainer, and runner settings.
    '''

    # init run io folder tree
    session_paths = artifacts.ResultsPaths(f'{config.exp_root}/results')
    session_paths.init()

    # save running config per run
    ctrl = artifacts.Controller[dict](session_paths.config) # generic, no policy
    ctrl.persist(dataclasses.asdict(config))

    # create a centralized main logger
    logger = utils.Logger('main', session_paths.main_log_file)

    # collect artifacts and build dataspsec
    artifact_paths=artifacts.ArtifactPaths(f'{config.exp_root}/artifacts')
    dataspecs = geopipe.build_dataspec(
        artifact_paths,
        ids_domain_name=config.dataspecs.domain_ids_name,
        vec_domain_name=config.dataspecs.domain_vec_name,
        print_out=True
    )

    # setup the model
    model = models.build_multihead_unet(
        dataspecs=dataspecs,
        backbone_config=config.models.body_registry[config.models.use_body],
        conditioning=config.models.conditioning,
        enable_logit_adjust=config.models.flags.enable_logit_adjust,
        enable_clamp=config.models.flags.enable_clamp,
        clamp_range=config.models.clamp_range
    )

    # build session runner
    runner = session.build_session_runner(
        dataspecs,
        model,
        config.session,
        session_paths,
        device='cuda' if torch.cuda.is_available() else 'cpu',
        logger=logger
    )

    # run session
    runner.fit()
