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
import typing
import uuid
# local imports
import landseg._constants as c
import landseg.artifacts as artifacts
import landseg.configs as configs
import landseg.execution.preflight.schema as schema


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
    result_list = (
        [results] if isinstance(results, schema.PreflightResult) else results
    )
    lines: list[str] = []
    width = 80
    sep = '=' * width
    thin_sep = '-' * width

    for result in result_list:
        lines.append(sep)
        header = f'PRE-FLIGHT READINESS CHECK: {result.target}'
        lines.append(f'{header:^{width}}')
        lines.append(sep)
        lines.append(
            f' {"CATEGORY":<11}{"PROBE ID":<26}{"STATUS":<9}{"DETAILS"}'
        )
        lines.append(thin_sep)

        for probe in result.probes:
            msg = probe.message
            if len(msg) > 32:
                msg = msg[:29] + '...'
            lines.append(
                f' {probe.category:<11}{probe.probe_id:<26}'
                f'{probe.status.value:<9}{msg}'
            )

        lines.append(sep)
        err_count = len(result.errors)
        warn_count = len(result.warnings)
        lines.append(
            f' STATUS: {result.status} '
            f'({err_count} errors, {warn_count} warnings)'
        )
        if result.telemetry:
            telemetry_str = ', '.join(
                f'{k}: {v}' for k, v in result.telemetry.items()
            )
            lines.append(f' Telemetry: {telemetry_str}')
        lines.append(sep)

    if len(result_list) > 1:
        ready_count = sum(1 for r in result_list if r.is_ready)
        blocked_count = len(result_list) - ready_count
        lines.append(
            f' SYSTEM STATUS: {ready_count} READY | {blocked_count} BLOCKED'
        )
        lines.append(sep)

    return '\n'.join(lines)


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
    result_list = (
        [results] if isinstance(results, schema.PreflightResult) else results
    )
    effective_exp_root = exp_root or root_config.execution.exp_root
    preflight_dir = os.path.join(effective_exp_root, 'preflight')

    now = datetime.datetime.now()
    timestamp_iso = now.strftime(c.TF_ISO8601)
    timestamp_tag = now.strftime('%Y%m%d_%H%M%S')
    uid = f'{timestamp_tag}_{uuid.uuid4().hex[:6]}'

    custom_report_path = root_config.command.preflight.report_path
    if custom_report_path:
        report_fp = custom_report_path
        parent_dir = os.path.dirname(report_fp)
        if parent_dir:
            os.makedirs(parent_dir, exist_ok=True)
    else:
        os.makedirs(preflight_dir, exist_ok=True)
        report_fp = os.path.join(
            preflight_dir, f'preflight_report_{uid}.json'
        )

    all_ready = all(r.is_ready for r in result_list)
    overall_status = 'READY' if all_ready else 'BLOCKED'

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
    total_probes = pass_count + warn_count + fail_count + skip_count

    report_payload: dict[str, typing.Any] = {
        'timestamp': timestamp_iso,
        'uid': uid,
        'target': target,
        'status': overall_status,
        'strict': bool(root_config.command.preflight.strict),
        'is_ready': all_ready,
        'summary': {
            'total_probes': total_probes,
            'pass': pass_count,
            'warn': warn_count,
            'fail': fail_count,
            'skip': skip_count,
        },
        'targets': [r.as_dict() for r in result_list],
    }

    report_ctrl = artifacts.Controller[dict](report_fp)
    report_ctrl.persist(report_payload)

    return report_fp, uid
