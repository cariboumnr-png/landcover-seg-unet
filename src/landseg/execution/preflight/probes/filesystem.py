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
Storage and filesystem diagnostic probes.

Validates output directory access, writability, and path readiness.

Public APIs:
    - `probe_storage`: Validate pipeline output directory access.
'''

# standard imports
import os
# local imports
# import landseg.artifacts as artifacts
# import landseg.configs as configs
import landseg.execution.preflight.schema as schema


# ----- public functions
def dir_writable(
    dir_path: str,
    pid: str | None = None
) -> schema.ProbeResult:
    '''Check if destination directory is writable.'''
    if os.path.exists(dir_path):
        check_dir = dir_path
    else:
        check_dir = os.path.dirname(dir_path)
    w_ok = os.access(check_dir, os.W_OK)
    status = schema.ProbeStatus.PASS if w_ok else schema.ProbeStatus.FAIL
    flag = 'is' if w_ok else 'is not'

    return schema.ProbeResult(
        pid=pid or 'target_directory',
        category='Filesystem',
        status=status,
        message=f'Target directory {flag} writable',
        details={'target_dir': check_dir},
    )


def file_exists(
    file_path: str,
    pid: str | None = None
) -> schema.ProbeResult:
    '''Check if the target file already exists.'''
    is_file = os.path.exists(file_path) and os.path.isfile(file_path)
    status = schema.ProbeStatus.WARN if is_file else schema.ProbeStatus.PASS
    flag = 'already' if is_file else 'does not'

    return schema.ProbeResult(
        pid=pid or 'target_file',
        category='Filesystem',
        status=status,
        message=f'Target file {flag} exists',
        details={'target_file_path': file_path},
    )
