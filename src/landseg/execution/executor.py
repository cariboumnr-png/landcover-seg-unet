# =========================================================================== #
#           Copyright (c) His Majesty the King in right of Ontario,           #
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
Pipeline execution
'''

# standard imports
import dataclasses
import os
import sys
import typing
# local imports
import landseg.artifacts as artifacts
import landseg.configs as configs
import landseg.execution.pipelines as piplines

DictControl = artifacts.Controller[dict[str, typing.Any]]

# -------------------------------Public Function------------------------------
def execute_pipeline(root_config: configs.RootConfig) -> typing.Any:
    '''Run the selected CLI pipeline with resolved configuration.'''

    # get running pipeline
    pipeline_name = root_config.pipeline.name

    # upstream detection checks
    _validate_upstream_pipelines(root_config, pipeline_name)

    # config staleness checks (if in CLI mode)
    _check_config_staleness(root_config, pipeline_name)

    # get command from pipeline
    command = piplines.get(pipeline_name)
    # run command and return result
    return command(root_config)

# -------------------------------private functions------------------------------
def _validate_upstream_pipelines(
    config: configs.RootConfig,
    pipeline_name: str
) -> None:
    '''Verify if upstream pipelines have completed successfully.'''

    # no checks if at the start of the pipeline chain
    if pipeline_name in ('default', 'data-ingest'):
        return

    # fetch data pipeline artifacts paths
    foundation_paths = artifacts.FoundationPaths(config.foundation.output_dpath)
    transform_paths = artifacts.TransformPaths(config.transform.output_dpath)

    # pipelines downstream of data-ingest
    ctrl_ingest = DictControl.load_json_or_fail(foundation_paths.report)
    try:
        report = ctrl_ingest.fetch()
        assert report # typing guard
    except artifacts.ArtifactError as e:
        raise artifacts.ArtifactError(
            'Upstream pipeline "data-ingest" has not been executed yet. '
            f'Missing or invalid ingestion report at canonical path: '
            f'{foundation_paths.report}'
        ) from e

    if report.get('status') != 'SUCCESS':
        status_val = report.get('status')
        raise artifacts.ArtifactError(
            'Upstream pipeline "data-ingest" status is '
            f'"{status_val}", not "SUCCESS". '
            'Please re-run "data-ingest" successfully first.'
        )

    # pipelines downstream of data-prepare
    if pipeline_name != 'data-prepare':
        ctrl_prep = DictControl.load_json_or_fail(transform_paths.report)
        try:
            report = ctrl_prep.fetch()
            assert report # typing guard
        except artifacts.ArtifactError as e:
            raise artifacts.ArtifactError(
                'Upstream pipeline "data-prepare" has not been executed yet. '
                f'Missing or invalid preparation report at canonical path: '
                f'{transform_paths.report}'
            ) from e

        if report.get('status') != 'SUCCESS':
            status_val = report.get('status')
            raise artifacts.ArtifactError(
                'Upstream pipeline "data-prepare" status is '
                f'"{status_val}", not "SUCCESS". '
                'Please re-run "data-prepare" successfully first.'
            )

def _check_config_staleness(
    config: configs.RootConfig,
    pipeline: str
) -> None:
    '''Check if active configs match those from previous runs.'''

    # staleness checks apply only in CLI mode and for dependent pipelines
    if not config.execution.cli_mode or pipeline in ('default', 'data-ingest'):
        return

    # fetch data pipeline artifacts paths
    foundation_paths = artifacts.FoundationPaths(config.foundation.output_dpath)
    transform_paths = artifacts.TransformPaths(config.transform.output_dpath)

    # initialize difference tracker
    diffs = {}

    # compare foundation configuration
    diffs.update(
        _compare_config_section(
            foundation_paths.config,
            'foundation',
            config.foundation,
        )
    )

    # compare transform configuration
    if pipeline != 'data-prepare':
        diffs.update(
            _compare_config_section(
                transform_paths.config,
                'transform',
                config.transform,
            )
        )

    if not diffs:
        return
    # display warning message
    print('\n' + '=' * 80)
    print(
        '[WARNING] Active configuration does not match settings '
        'used to build data artifacts:'
    )
    for path, (active, saved) in diffs.items():
        print(f'  - {path}: active={active} vs recorded={saved}')
    print('=' * 80)

    # check if stdin is a TTY for interactive confirmation
    if sys.stdin.isatty():
        sys.stdout.write(
            '\nStale artifacts detected. Do you want to proceed '
            'with execution anyway? [y/N]: '
        )
        sys.stdout.flush()
        response = sys.stdin.readline().strip().lower()
        if response not in ('y', 'yes'):
            print('Execution aborted by user due to config mismatch.')
            sys.exit(1)
    else:
        print(
            '\nNon-interactive environment detected. '
            'Proceeding execution with warning.'
        )
        print('=' * 80 + '\n')

def _compare_config_section(
    artifact_path: str,
    section_name: str,
    current: typing.Any,
) -> dict[str, tuple[typing.Any, typing.Any]]:
    '''Helper to compare a config section'''

    ctrl = DictControl.load_json_or_fail(artifact_path)

    try:
        saved = ctrl.fetch()          # or fetch()+check
        assert saved # typing
    except artifacts.ArtifactError:
        return {}

    try:
        saved_section = _normalize_val(saved.get(section_name, {}))
        current_section = _normalize_val(dataclasses.asdict(current))
    except (TypeError, AttributeError):
        return {f'{section_name}.config_file_read': (True, False)}

    return _diff_configs(
        current_section,
        saved_section,
        section_name,
    )

def _normalize_val(val: typing.Any) -> typing.Any:
    '''Normalize values for robust config comparison'''

    if isinstance(val, dict):
        return {k: _normalize_val(v) for k, v in val.items()}
    if isinstance(val, list):
        return [_normalize_val(v) for v in val]
    if isinstance(val, str):
        # normalize slashes and paths
        if '/' in val or '\\' in val or val.startswith('.'):
            return os.path.abspath(val).replace('\\', '/')
        return val
    return val

def _diff_configs(
    dict1: dict,
    dict2: dict,
    prefix: str = ''
) -> dict[str, tuple[typing.Any, typing.Any]]:
    '''Recursively compare two normalized configuration dictionaries.'''

    diffs = {}
    for k, v in dict1.items():
        path = f'{prefix}.{k}' if prefix else k
        if k not in dict2:
            diffs[path] = (v, None)
            continue

        v2 = dict2[k]
        if isinstance(v, dict) and isinstance(v2, dict):
            diffs.update(_diff_configs(v, v2, path))
        elif isinstance(v, list) and isinstance(v2, list):
            if len(v) != len(v2):
                diffs[path] = (v, v2)
            else:
                for i, (item1, item2) in enumerate(zip(v, v2)):
                    list_path = f'{path}[{i}]'
                    if isinstance(item1, dict) and isinstance(item2, dict):
                        diffs.update(_diff_configs(item1, item2, list_path))
                    elif item1 != item2:
                        diffs[list_path] = (item1, item2)
        else:
            if v != v2:
                diffs[path] = (v, v2)

    for k, v2 in dict2.items():
        path = f'{prefix}.{k}' if prefix else k
        if k not in dict1:
            diffs[path] = (None, v2)

    return diffs
