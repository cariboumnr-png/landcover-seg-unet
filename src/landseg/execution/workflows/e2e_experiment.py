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
Full experiment execution workflow.

Orchestrates the entire geospatial machine learning lifecycle from
world grid generation through harmonization, ingestion, data preparation,
and model training.

Public APIs:
    - `execute_e2e_experiment`: run full experiment lifecycle.
'''

# local imports
import landseg.configs as configs
import landseg.execution.pipelines as pipelines
import landseg.utils as utils


# ----- public functions
def execute_e2e_experiment(config: configs.RootConfig) -> None:
    '''
    Execute full end-to-end experiment pipeline sequence.

    Runs world grid generation, data harmonization, batch ingestion,
    data preparation, and model training in a unified sequence.

    Args:
        config:
            Root execution configuration.
    '''
    logger = utils.Logger(name='e2e-experiment', enable_file_log=False)
    logger.log_sep()
    logger.info('Starting Full End-to-End Experiment workflow sequence...')
    logger.log_sep()

    # stage 1: canonical world grid
    logger.info('[1/5] Stage: World Grid Generation')
    pipelines.WorldGridGeneration(config).run()

    # stage 2: data harmonization
    logger.info('[2/5] Stage: Data Harmonization')
    pipelines.DataHarmonization(config).run()

    # stage 3: batch ingestion into canonical block pool
    logger.info('[3/5] Stage: Batch Ingestion')
    pipelines.DataIngestion(config).run()

    # stage 4: data preparation (AOI partition splits & DataSpecs)
    logger.info('[4/5] Stage: Data Preparation')
    pipelines.DataPreparation(config).run()

    # stage 5: neural model training session
    logger.info('[5/5] Stage: Model Training')
    pipelines.ModelTraining(config).run()

    logger.log_sep()
    logger.info('Full End-to-End Experiment workflow completed successfully.')
    logger.log_sep()
    logger.close()
