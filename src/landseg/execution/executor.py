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
Pipeline execution
'''

# standard imports
import typing
# local imports
import landseg.execution.pipelines as piplines
import landseg.configs as configs

#
def execute_pipeline(root_config: configs.RootConfig) -> typing.Any:
    '''Run the selected CLI pipeline with resolved configuration.'''

    # get running pipeline
    pipeline_name = root_config.pipeline.name
    # get command from pipeline
    command = piplines.get(pipeline_name)
    # run command and return result
    return command(root_config)
