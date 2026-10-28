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
Domain handlers and contract validators for data harmonization.

Provides authoritative validation routines and execution wrappers for
dataset manifest compilation, input compatibility checks, and
raster harmonization execution.

Public APIs:
    - `validate_grid_reference`: Load and validate world grid reference.
    - `validate_manifest_contract`: Validate manifest and output uniqueness.
'''

# local imports
import landseg.geopipe.harmonize.context as context
import landseg.geopipe.harmonize.manifest.compiler as compiler


# ----- public functions
validate_grid_reference = context.load_grid_reference
validate_manifest_contract = compiler.compile_dataset_manifest
