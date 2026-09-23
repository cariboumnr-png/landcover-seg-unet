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
import landseg.geopipe.core as geo_core

# aliases
DictControl = artifacts.Controller[dict[str, typing.Any]]


# ----- public functions
def execute_pipeline(root_config: configs.RootConfig) -> typing.Any:
    '''Run the selected CLI pipeline with resolved configuration.'''
    # upstream detection checks
    _validate_upstream_pipelines(root_config)
    # get command from pipeline
    command = piplines.get(root_config.pipeline.name)
    # run command and return result
    return command(root_config)


# ----- private helpers
def _validate_upstream_pipelines(config: configs.RootConfig) -> None:
    '''Verify if upstream pipelines have completed successfully.'''
    # get running pipeline
    pipeline = config.pipeline.name

    # no checks if at the start of the pipeline chain
    if pipeline in ('default', 'world-grid'):
        return

    # check world-grid status if running data-harmonize
    if pipeline == 'data-harmonize':
        report_fp = geo_core.get_grid_report_fpath(
            config.data.world_grid.output_dpath
        )
        p = config.data.world_grid.params
        gid = geo_core.GridLayout.generate_gid(p.tile_size, p.tile_stride)
        grid_fpath = os.path.join(
            config.data.world_grid.output_dpath, f'{gid}.json'
        )
        if not os.path.exists(report_fp) and not os.path.exists(grid_fpath):
            raise artifacts.ArtifactError(
                'Upstream pipeline "world-grid" has not been executed yet. '
                f'Missing world grid report or artifact at canonical path: '
                f'{report_fp}'
            )
        return

    # fetch data pipeline artifacts paths
    art_paths = artifacts.ArtifactPaths.from_config(config)
    paths_harmonize = art_paths.data_harmonization
    paths_ingest = art_paths.data_ingestion
    paths_prepare = art_paths.data_preparation

    # locate targeted/latest harmonization run folder if present
    try:
        paths_harmonize.get_run_folder(config.data.ingestion.harmonization_run)
    except FileNotFoundError:
        pass # fallback to default folder

    # artifacts controllers (after harmonization run folder location)
    ctrl_harmonize = DictControl.load_json_or_fail(paths_harmonize.report)
    ctrl_ingest = DictControl.load_json_or_fail(paths_ingest.report)
    ctrl_prep = DictControl.load_json_or_fail(paths_prepare.report)

    # check data-harmonize status if running data-ingest
    if pipeline == 'data-ingest':

        # fetch data harmonization report
        try:
            report_harmonize = ctrl_harmonize.fetch()
        except artifacts.ArtifactError as e:
            raise artifacts.ArtifactError(
                'Upstream pipeline "data-harmonize" has not been executed yet.'
                f' Missing or invalid harmonization report at canonical path: '
                f'{paths_harmonize.report}'
            ) from e

        # check data-harmonize report status
        if report_harmonize.get('status') != 'SUCCESS':
            status_val = report_harmonize.get('status')
            raise artifacts.ArtifactError(
                'Upstream pipeline "data-harmonize" status is '
                f'"{status_val}", not "SUCCESS". '
                'Please re-run "data-harmonize" successfully first.'
            )

        # check existing vs configured source of harmonized data
        try:
            report_ingest = ctrl_ingest.fetch()

            if report_ingest.get('status') != 'SUCCESS':
                return # existing ingestion not successful, proceed

            source = report_harmonize.get('finalized_rasters')
            assert source, 'Invalid harmonized data source'
            canonical_blocks = report_ingest.get('data_blocks', {})
            image_fp = canonical_blocks.get('image_filepath')
            label_fp = canonical_blocks.get('label_filepath')
            if (
                source.get('features') == image_fp
                and source.get('labels') == label_fp
                and not config.data.ingestion.rebuild
            ):
                print('\n' + '=' * 80)
                print(
                    f'[WARNING] Ingesting the same harmonized data source '
                    f'as recorded in: {paths_harmonize.report}\n'
                    f'[NOTE] Current ingestion "rebuild" flag is set to '
                    f'[{config.data.ingestion.rebuild}]\n'
                )
                _user_y_n_popup()

        except artifacts.ArtifactError:
            return # no ingestion has been run yet, proceed

    elif pipeline == 'data-prepare':

        # ingestion reports are now telemetry, e.g., ./run_0001/report.json ...
        # data-prepare pipeline no longer depends on a particular ingestion run
        # instead it requires valid catalog.json and schema.json
        # which is validated during `build_preparation_context()`

        # check existing data-prepare report
        try:
            report_prep = ctrl_prep.fetch()

            if report_prep.get('status') != 'SUCCESS':
                return # existing preparation not successful, proceed

            # evaluate differences against recorded prep config
            ctrl = DictControl.load_json_or_fail(paths_prepare.config)
            try:
                saved_config = ctrl.fetch()
            except artifacts.ArtifactError:
                print(f'Error reading config at: {paths_prepare.config}')
                raise

            current_config = dataclasses.asdict(config.data.preparation)
            if (
                saved_config['data']['preparation'] == current_config
                and not config.data.preparation.rebuild
            ):

                print('\n' + '=' * 80)
                print(
                    f'[WARNING] Preparing data using the same configuration '
                    f'as recorded in: {paths_prepare.config}\n'
                    f'[NOTE] Current preparation "rebuild" flag is set to '
                    f'[{config.data.preparation.rebuild}]\n'
                )
                _user_y_n_popup()

        except artifacts.ArtifactError:
            return # no preparation has been run yet, proceed

    # pipelines downstream of data-prepare, e.g., session running
    else:

        try:
            report = ctrl_prep.fetch()
        except artifacts.ArtifactError as e:
            raise artifacts.ArtifactError(
                'Upstream pipeline "data-prepare" has not been executed yet. '
                f'Missing or invalid preparation report at canonical path: '
                f'{paths_prepare.report}'
            ) from e

        if report.get('status') != 'SUCCESS':
            status_val = report.get('status')
            raise artifacts.ArtifactError(
                'Upstream pipeline "data-prepare" status is '
                f'"{status_val}", not "SUCCESS". '
                'Please re-run "data-prepare" successfully first.'
            )


def _user_y_n_popup():
    '''Console whether to proceed pop-up from user input.'''
    # check if stdin is a TTY for interactive confirmation
    if sys.stdin.isatty():
        sys.stdout.write('Proceed anyways [y/N]:')
        sys.stdout.flush()
        response = sys.stdin.readline().strip().lower()
        if response not in ('y', 'yes'):
            print('Execution aborted by user.')
            sys.exit(1)
    else:
        print('\nNon-interactive env. detected. Proceed with warning.')
        print('=' * 80 + '\n')
