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
Multi-batch ingestion workflow configurator.

Public APIs:
    - `BatchIngestConfigurator`: Configure multi-batch ingestion workflows.
'''

# local imports
import landseg.adapters.api.configurators.data_ingest as data_ingest


# ----- public classes
class BatchIngestConfigurator(data_ingest.DataIngestionConfigurator):
    '''Configure multi-batch data ingestion workflow.'''

    def __init__(
        self,
        experiment_root: str,
    ):
        super().__init__(experiment_root)
        self._cfg.command.name = 'batch-ingest'
