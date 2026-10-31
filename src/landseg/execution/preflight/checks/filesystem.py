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
Filesystem and storage diagnostic checks.

Validates output directory access, writability, and path readiness.

Public APIs:
    - `dir_writable`: Validate a pipeline output directory is writable.
'''

# standard imports
import os
import pathlib
# local imports
import landseg.execution.preflight.schema as schema


# ----- public functions
def dir_writable(
    dir_path: pathlib.Path | None,
    pid: str,
) -> schema.ProbeResult:
    '''Check that a destination directory chain is writable.'''
    if dir_path is None:
        return schema.ProbeResult(
            pid=pid,
            category=schema.ProbeCategory.FILESYSTEM,
            status=schema.ProbeStatus.FAIL,
            message='Target directory path is not defined',
            details={'target_dir': None},
        )

    # walk up to the nearest existing ancestor
    check_path = dir_path.resolve()
    while not check_path.exists():
        parent = check_path.parent
        if parent == check_path:
            break
        check_path = parent

    is_writable = os.access(check_path, os.W_OK)
    return schema.ProbeResult(
        pid=pid,
        category=schema.ProbeCategory.FILESYSTEM,
        status=(
            schema.ProbeStatus.PASS
            if is_writable
            else schema.ProbeStatus.FAIL
        ),
        message=(
            f'Target directory {"is" if is_writable else "is not"} writable'
        ),
        details={'target_dir': str(dir_path), 'checked': str(check_path)},
    )
