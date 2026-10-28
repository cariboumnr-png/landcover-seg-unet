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
Domain handlers and contract validators for data ingestion.

Provides authoritative validation routines and execution wrappers for
raster alignment, feature engineering specifications, and pool grid
compatibility.

Public APIs:
    - `validate_block_feature_specs`: Validate feature engineering specs.
    - `validate_pool_grid_compatibility`: Validate pool grid alignment.
    - `validate_raster_alignment`: Validate raster CRS and resolution.
'''

# local imports
import landseg.geopipe.ingest.blocks.assembler.builder as builder
import landseg.geopipe.ingest.blocks.mapper.mapper as mapper
import landseg.geopipe.ingest.context as context


# ----- public functions
validate_block_feature_specs = builder.validate_block_feature_specs
validate_pool_grid_compatibility = context.verify_pool_grid_compatibility
validate_raster_alignment = mapper.validate_raster_alignment
