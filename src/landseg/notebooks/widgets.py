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
Interactive parameter widget abstraction for notebook environments.

Compatibility shim delegating to `landseg.runtime.interactive`.

Public APIs:
    - `MissingRequiredInputError`: Error on missing notebook parameters.
    - `NotebookWidgets`: Ergonomic widget wrapper across runtimes.
    - `require_widget`: Enforce and validate required notebook widgets.
'''

# local imports
from landseg.runtime import interactive

# re-exports
MissingRequiredInputError = interactive.MissingRequiredInputError
NotebookWidgets = interactive.NotebookWidgets
require_widget = interactive.require_widget

__all__ = [
    'MissingRequiredInputError',
    'NotebookWidgets',
    'require_widget',
]
