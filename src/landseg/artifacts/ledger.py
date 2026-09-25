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
Run ledger utilities and deterministic payload fingerprinting.

Provides centralized helpers for computing canonical cryptographic
fingerprints of run specifications and detecting collision with
completed runs in pipeline run manifests.

Public APIs:
    - `compute_fingerprint`: Deterministic SHA-256 hash of payload.
    - `check_run_collision`: Search runs manifest for matching run.
'''

# standard imports
from __future__ import annotations
import hashlib
import json
import os
import typing
# local imports
import landseg.artifacts.controller as controller


# ----- public functions
def compute_fingerprint(payload: typing.Any) -> str:
    '''
    Compute deterministic SHA-256 fingerprint from serializable data.

    Serializes the input payload with sorted JSON keys to guarantee a
    canonical string representation before computing the digest.

    Args:
        payload:
            JSON-serializable data structure (dict, list, primitive).

    Returns:
        str:
            Hexadecimal SHA-256 digest string.
    '''
    canonical_repr = json.dumps(payload, sort_keys=True)
    return hashlib.sha256(canonical_repr.encode('utf-8')).hexdigest()


def check_run_collision(
    fingerprint: str,
    runs_manifest_fpath: str,
    *,
    success_status: str = 'SUCCESS',
) -> str | None:
    '''
    Check whether an existing successful run matches the fingerprint.

    Inspects the runs ledger manifest on disk if present. Looks for an
    entry whose fingerprint matches and status equals success_status.

    Args:
        fingerprint:
            Deterministic SHA-256 hash string of the run identity.
        runs_manifest_fpath:
            File path to the runs manifest JSON ledger.
        success_status:
            Expected status string representing an active completed run.

    Returns:
        str | None:
            Colliding run identifier if found, otherwise None.
    '''
    if not os.path.exists(runs_manifest_fpath):
        return None

    ctrl = controller.Controller[dict[str, dict[str, typing.Any]]](
        runs_manifest_fpath
    )
    manifest = ctrl.fetch() or {}

    for run_uid, record in manifest.items():
        if (
            record.get('fingerprint') == fingerprint
            and record.get('status') == success_status
        ):
            return run_uid

    return None
