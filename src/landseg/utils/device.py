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
Hardware device resolution and query utilities.

Compatibility shim delegating to `landseg.runtime.hardware`.

Public APIs:
    - `get_device_name`: Query human-readable name of execution device.
    - `is_cpu_bf16_supported`: Check CPU hardware bfloat16 capability.
    - `resolve_device`: Resolve hardware execution target string.
'''

# local imports
from landseg.runtime import hardware

# re-exports
get_device_name = hardware.get_device_name
is_cpu_bf16_supported = hardware.is_cpu_bf16_supported
resolve_device = hardware.resolve_device

__all__ = [
    'get_device_name',
    'is_cpu_bf16_supported',
    'resolve_device',
]
