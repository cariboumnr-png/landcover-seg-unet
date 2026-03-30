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
Dev Test Ground
'''

# local imports
import landseg.configs as configs
import landseg.factory as factory
import landseg.geopipe.specification as specification
import landseg.models as models
import landseg.utils as utils

def train(config: configs.RootConfig):
    '''test'''

    artifacts_root = './experiment/artifacts/'
    logger = utils.Logger('test', './test.log')
    # build dataspsec
    dataspecs = specification.build_dataspec(
        f'{artifacts_root}/foundation/model_dev/metadata.json',
        f'{artifacts_root}/domain_knowledge/ecodistrict.json',
        f'{artifacts_root}/domain_knowledge/geology.json',
        f'{artifacts_root}/transform/schema.json',
    )
    print(dataspecs)

    # setup the model
    model = models.build_multihead_unet(dataspecs, config.models)

    # build controller
    exp_dir = './experiment'
    runner = factory.build_runner(exp_dir, dataspecs, model, config, logger)

    # run via controller
    runner.fit()
