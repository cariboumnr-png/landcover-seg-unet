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
Programmatic API entry
'''

# local imports
import landseg.configs as configs
import landseg.execution as execution
import landseg.utils as utils

def run(root_config: configs.RootConfig):
    '''Run pipeline'''

    logger = utils.Logger('api', './api.log')
    try:
        logger.log('INFO', f'Running pipeline: {root_config.pipeline.name}')
        return execution.execute_pipeline(root_config)
    except KeyboardInterrupt:
        logger.log('INFO', 'Execution interrupted')
        raise
    except Exception:
        logger.log(
            'CRITICAL',
            'Unhandled exception occurred during API execution',
            exc_info=True,
        )
        raise
