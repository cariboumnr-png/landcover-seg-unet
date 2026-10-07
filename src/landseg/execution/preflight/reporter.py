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
Terminal formatting and report artifact export for pre-flight readiness.

Provides ASCII dashboard rendering for console output and flat JSON
artifact persistence under the experiment preflight directory.

Public APIs:
    - `format_preflight_report`: Render formatted terminal dashboard.
    - `export_preflight_report`: Persist pre-flight JSON artifact.
'''

# standard imports
import datetime
import os
import uuid
# local imports
import landseg._constants as c
import landseg.artifacts as artifacts
import landseg.configs as configs
import landseg.execution.preflight.schema as schema


# constants
MAX_WIDTH = 120
HEADER_ROW = f' {"CATEGORY":<15}{"PROBE ID":<30}{"STATUS":<9}{"DETAILS"}'
MSG_MAX_LEN = MAX_WIDTH - 15 - 30 - 9 - 3 # 3 as the len of "..."

# ----- public functions
def format_preflight_report(
    results: schema.PreflightResult | list[schema.PreflightResult],
) -> str:
    '''
    Format pre-flight inspection results into a terminal status dashboard.

    Args:
        results:
            single PreflightResult or list of PreflightResult instances.

    Returns:
        str:
            formatted terminal dashboard text.
    '''
    result_list = results if isinstance(results, list) else [results]
    lns: list[str] = []
    sep = '=' * MAX_WIDTH
    thin_sep = '-' * MAX_WIDTH

    for result in result_list:
        lns.append(sep)
        header = f'PRE-FLIGHT READINESS CHECK: {result.target}'
        lns.append(f'{header:^{MAX_WIDTH}}')
        lns.append(sep)
        lns.append(HEADER_ROW)
        lns.append(thin_sep)

        for p in result.probes:
            m = p.message
            if len(m) > MSG_MAX_LEN:
                m = m[:MSG_MAX_LEN - 1] + '...'
            lns.append(f' {p.category:<15}{p.pid:<30}{p.status.value:<9}{m}')

        lns.append(sep)

        if result.telemetry:
            lns.append(' Telemetry:')
            for k, v in result.telemetry.items():
                lns.append(f'   - {k:<15}\t{v}')

        e_n = len(result.errors)
        w_n = len(result.warnings)
        lns.append(f' STATUS: {result.status} ({e_n} errors, {w_n} warnings)')

        lns.append(sep)

    if len(result_list) > 1:
        ready_n = sum(1 for r in result_list if r.is_ready)
        blocked_n = len(result_list) - ready_n
        lns.append(f' SYSTEM STATUS: {ready_n} READY | {blocked_n} BLOCKED')
        lns.append(sep)

    return '\n'.join(lns)


def export_preflight_report(
    results: schema.PreflightResult | list[schema.PreflightResult],
    root_config: configs.RootConfig,
    target: str,
    *,
    exp_root: str | None = None,
) -> tuple[str, str]:
    '''
    Export pre-flight report to a timestamped JSON artifact.

    Args:
        results:
            single PreflightResult or list of PreflightResult instances.
        root_config:
            hydra-composed root configuration.
        target:
            selected target pipeline identifier.
        exp_root:
            optional experiment root directory override.

    Returns:
        tuple[str, str]:
            persisted report file path and generated timestamped UID.
    '''
    result_list = results if isinstance(results, list) else [results]

    now = datetime.datetime.now()
    uid = f'{now.strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}'

    if root_config.command.preflight.report_path:
        report_fp = root_config.command.preflight.report_path
        report_dir = os.path.dirname(report_fp)
    else:
        effective_exp_root = exp_root or root_config.execution.exp_root
        report_dir = os.path.join(effective_exp_root, 'preflight')
        report_fp = os.path.join(report_dir, f'preflight_report_{uid}.json')

    if report_dir:
        os.makedirs(report_dir, exist_ok=True)

    all_ready = all(r.is_ready for r in result_list)

    pass_count = sum(
        sum(1 for p in r.probes if p.status == schema.ProbeStatus.PASS)
        for r in result_list
    )
    warn_count = sum(
        sum(1 for p in r.probes if p.status == schema.ProbeStatus.WARN)
        for r in result_list
    )
    fail_count = sum(
        sum(1 for p in r.probes if p.status == schema.ProbeStatus.FAIL)
        for r in result_list
    )
    skip_count = sum(
        sum(1 for p in r.probes if p.status == schema.ProbeStatus.SKIP)
        for r in result_list
    )

    artifacts.Controller[dict](report_fp).persist({
        'timestamp': now.strftime(c.TF_ISO8601),
        'uid': uid,
        'target': target,
        'status': 'READY' if all_ready else 'BLOCKED',
        'strict': bool(root_config.command.preflight.strict),
        'is_ready': all_ready,
        'summary': {
            'total_probes': pass_count + warn_count + fail_count + skip_count,
            'pass': pass_count,
            'warn': warn_count,
            'fail': fail_count,
            'skip': skip_count,
        },
        'targets': [r.as_dict() for r in result_list],
    })

    return report_fp, uid
