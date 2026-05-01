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

# pylint: disable=no-value-for-parameter

'''
CLI entry.
'''

# standard imports
import sys
import typing
# third-party imports
import hydra
import omegaconf
# local imports
import landseg.execution as execution
import landseg.utils as utils

# main process
@hydra.main('pkg://landseg/configs', 'config', version_base='1.3')
def main(config: omegaconf.DictConfig) -> typing.Any:
    '''Run the selected CLI pipeline with resolved configuration.'''

    # cli logger
    logger = utils.Logger('cli', './cli.log')

    # run specified mode with exceptions handling
    try:
        root_config = execution.resolve_configs(config)
        return execution.execute_pipeline(root_config)
    # manual keyboard interruption
    except KeyboardInterrupt:
        logger.log('INFO', '\nExperiment manually interrupted, exiting...')
        sys.exit(130)
    # capture others and log
    except Exception: # pylint: disable=broad-exception-caught
        logger.log('CRITICAL', 'Unhandled exception occurred', exc_info=True)
        sys.exit(1)

if __name__ == '__main__':
    main()
