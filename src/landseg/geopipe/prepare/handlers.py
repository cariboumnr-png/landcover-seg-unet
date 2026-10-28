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
Domain handlers and contract validators for data preparation.

Provides authoritative validation routines and execution wrappers for
upstream preparation inputs, dataset view resolution, feature channels,
and target head hierarchies.

Public APIs:
    - `validate_dataset_view`: Validate and compile dataset view.
    - `validate_feature_channels`: Validate requested feature channels.
    - `validate_focal_head`: Validate focal target head selection.
    - `validate_preparation_context`: Validate preparation inputs.
    - `validate_target_heads`: Validate multi-head target hierarchy.
'''

# local imports
import landseg.geopipe.prepare.context as context
import landseg.geopipe.prepare.dataset.semantics as semantics
import landseg.geopipe.prepare.dataset.view as view


# ----- public functions
validate_dataset_view = view.build_dataset_view
validate_feature_channels = semantics.resolve_feature_channels
validate_focal_head = semantics.resolve_focal_head
validate_preparation_context = context.build_preparation_context
validate_target_heads = semantics.resolve_target_heads
