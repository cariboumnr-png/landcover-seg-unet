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
Pre-flight data structures and result schemas.

Defines diagnostic probe execution statuses, individual probe result
records, and aggregated pre-flight readiness results.

Public APIs:
    - `ProbeStatus`: Diagnostic probe status enumeration.
    - `ProbeResult`: Immutable evaluation record for a single probe.
    - `PreflightResult`: Aggregated validation result for a target.
'''

# standard imports
import dataclasses
import enum
import typing


# ----- public types
class ProbeStatus(enum.StrEnum):
    '''Diagnostic probe execution status.'''
    PASS = 'PASS'
    WARN = 'WARN'
    FAIL = 'FAIL'
    SKIP = 'SKIP'


# ----- public dataclasses
@dataclasses.dataclass(frozen=True)
class ProbeResult:
    '''Single diagnostic probe evaluation result.'''
    probe_id: str
    category: str
    status: ProbeStatus
    message: str
    details: dict[str, typing.Any] = dataclasses.field(default_factory=dict)


@dataclasses.dataclass
class PreflightResult:
    '''Aggregated pre-flight inspection result for a pipeline.'''
    target: str
    status: str
    probes: list[ProbeResult] = dataclasses.field(default_factory=list)
    telemetry: dict[str, typing.Any] = dataclasses.field(default_factory=dict)

    @property
    def is_ready(self) -> bool:
        '''Return True if no probes have failed.'''
        return all(p.status != ProbeStatus.FAIL for p in self.probes)

    @property
    def errors(self) -> list[str]:
        '''Return error messages from failed probes.'''
        return [p.message for p in self.probes if p.status == ProbeStatus.FAIL]

    @property
    def warnings(self) -> list[str]:
        '''Return warning messages from warning probes.'''
        return [p.message for p in self.probes if p.status == ProbeStatus.WARN]

    def as_dict(self) -> dict[str, typing.Any]:
        '''Return dictionary representation for report serialization.'''
        return {
            'target': self.target,
            'status': self.status,
            'is_ready': self.is_ready,
            'probes': [
                {
                    'probe_id': p.probe_id,
                    'category': p.category,
                    'status': p.status.value,
                    'message': p.message,
                    'details': p.details,
                }
                for p in self.probes
            ],
            'errors': self.errors,
            'warnings': self.warnings,
            'telemetry': self.telemetry,
        }
