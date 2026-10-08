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
Programmatic API entry
'''

# standard imports
import typing
# local imports
import landseg.configs as configs
import landseg.execution as execution
import landseg.execution.preflight as preflight
import landseg.utils as utils


# ----- public functions
def run(root_config: configs.RootConfig) -> typing.Any:
    '''
    Run execution pipeline or workflow with resolved configuration.

    Args:
        root_config:
            Root execution configuration instance.

    Returns:
        typing.Any:
            Result returned by the dispatched pipeline or workflow.
    '''
    logger = utils.Logger('api', './api.log')
    try:
        logger.info(f'Running command: {root_config.command.name}')
        return execution.execute_command(root_config)
    except KeyboardInterrupt:
        logger.info('Execution interrupted')
        raise
    except Exception:
        logger.exception('Unhandled exception occurred during API execution')
        raise


def run_preflight(
    config: configs.RootConfig | str | None = None,
    target: str | None = None,
    *,
    strict: bool | None = None,
    export_report: bool | None = None,
    report_path: str | None = None,
    check_gpu: bool | None = None,
    exp_root: str | None = None,
) -> preflight.PreflightResult | list[preflight.PreflightResult]:
    '''
    Run pre-flight diagnostic validation checks programmatically.

    Probes system resources, spatial grid alignments, upstream ledgers,
    and dataset integrity prior to committing heavy computation.

    Args:
        config:
            Optional root configuration or target string. Defaults to a
            new `RootConfig` instance.
        target:
            Optional pipeline or workflow target name (e.g.,
            `'model-train'`, `'batch-ingest'`, or `'all'`).
        strict:
            If specified, overrides `config.command.preflight.strict`.
        export_report:
            If specified, overrides `config.command.preflight.export_report`.
        report_path:
            If specified, overrides `config.command.preflight.report_path`.
        check_gpu:
            If specified, overrides `config.command.preflight.check_gpu`.
        exp_root:
            Optional experiment root directory override.

    Returns:
        preflight.PreflightResult | list[preflight.PreflightResult]:
            Pre-flight validation report or list of reports across targets.
    '''
    if isinstance(config, str) and target is None:
        resolved_target = config
        cfg = configs.RootConfig()
    else:
        cfg = config if isinstance(config, configs.RootConfig) else configs.RootConfig()
        resolved_target = target

    if strict is not None:
        cfg.command.preflight.strict = strict
    if export_report is not None:
        cfg.command.preflight.export_report = export_report
    if report_path is not None:
        cfg.command.preflight.report_path = report_path
    if check_gpu is not None:
        cfg.command.preflight.check_gpu = check_gpu

    if resolved_target is None:
        if cfg.command.name not in ('preflight', 'default'):
            resolved_target = cfg.command.name
        else:
            resolved_target = cfg.command.preflight.target

    return preflight.run_preflight(
        root_config=cfg,
        target=resolved_target,
        exp_root=exp_root,
    )


def run_intake(
    config: configs.RootConfig | None = None,
) -> typing.Any:
    '''
    Run continuous end-to-end data intake (harmonize + ingest).

    Args:
        config:
            Optional root execution configuration instance.

    Returns:
        typing.Any:
            Result returned by the dispatched intake workflow.
    '''
    cfg = config if isinstance(config, configs.RootConfig) else configs.RootConfig()
    cfg.command.name = 'e2e-intake'
    return run(cfg)


def run_experiment(
    config: configs.RootConfig | None = None,
) -> typing.Any:
    '''
    Run full lifecycle experiment (grid, harmonize, ingest, prepare, train).

    Args:
        config:
            Optional root execution configuration instance.

    Returns:
        typing.Any:
            Result returned by the dispatched experiment workflow.
    '''
    cfg = config if isinstance(config, configs.RootConfig) else configs.RootConfig()
    cfg.command.name = 'e2e-experiment'
    return run(cfg)
