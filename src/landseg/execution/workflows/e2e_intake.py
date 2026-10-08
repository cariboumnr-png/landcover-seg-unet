# =========================================================================== #
#            Copyright © His Majesty the King in right of Ontario,            #
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
Continuous end-to-end data intake workflow.

Coordinates raw raster harmonization followed immediately by batch
ingestion into the canonical block pool.

Public APIs:
    - `execute_e2e_intake`: run data intake sequence.
'''

# local imports
import landseg.configs as configs
import landseg.execution.pipelines as pipelines
import landseg.execution.workflows.batch_ingest as batch_ingest
import landseg.utils as utils


# ----- public functions
def execute_e2e_intake(config: configs.RootConfig) -> None:
    '''
    Execute continuous data harmonization and batch ingestion.

    Coordinates sequential execution of the data harmonization
    pipeline followed by batch ingestion into the canonical pool.

    Args:
        config:
            Root execution configuration containing data settings.
    '''
    logger = utils.Logger(name='e2e-intake', enable_file_log=False)
    logger.log_sep()
    logger.log('INFO', 'Starting End-to-End Data Intake workflow...')
    logger.log_sep()

    # stage 1: harmonize raw rasters onto world grid canvas
    logger.log('INFO', '[1/2] Running Data Harmonization pipeline...')
    pipelines.DataHarmonization(config).run()

    # stage 2: ingest pending harmonized batches into the block pool
    logger.log('INFO', '[2/2] Running Batch Ingestion workflow...')
    batch_ingest.execute_batch_ingest(config)

    logger.log_sep()
    logger.log(
        'INFO',
        'End-to-End Data Intake workflow completed successfully.'
    )
    logger.log_sep()
    logger.close()
