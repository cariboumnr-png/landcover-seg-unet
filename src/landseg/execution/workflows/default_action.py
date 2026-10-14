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
Default CLI action module.

Displays available workflow commands and usage summary when no explicit
command is provided.

Public APIs:
    - `execute_default_action`: Print help summary to console.
'''

# standard imports
import textwrap
# local imports
import landseg.configs as configs


# ----- public functions
def execute_default_action(config: configs.RootConfig) -> None:
    '''
    Print default CLI overview and usage instructions.

    Args:
        config:
            Root execution configuration instance.
    '''
    _ = config
    print(
        textwrap.dedent(
            '''
            LANDSEG

            Land-cover segmentation framework for geospatial data preparation,
            model training, evaluation, and study workflows.

            Commands:
            world-grid        Build canonical world grid
            data-harmonize    Harmonize source rasters
            data-ingest       Build data block artifacts
            data-prepare      Create datasets and metadata
            model-train       Train a model
            model-evaluate    Evaluate a checkpoint
            diagnose-overfit  Run overfit diagnostics
            study-sweep       Run Optuna studies
            study-analysis    Analyze study results
            preflight         Run readiness validation
            e2e-intake        Run continuous data intake workflow
            e2e-experiment    Run full 5-stage end-to-end experiment

            Example:
            landseg command=model-train
            '''
        )
    )
