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

# pylint: disable=protected-access

'''Unit tests for block image and label statistics aggregation.'''

# standard imports
import pathlib
# third-party imports
import numpy
import pytest
# local imports
import landseg.geopipe.contracts.preparation as contracts
import landseg.geopipe.prepare.materialize.stats as mat_stats
import landseg.utils.validator as val


# ----- `count_label` tests
def test_count_label_success():
    '''
    Given: Raw class counts mapping and selected block coordinate
        strings.
    When: Running count_label.
    Then: Correctly aggregate per-head class pixel counts.
    '''
    raw_counts = {
        (0, 0): {'head1': [10, 20], 'head2': [5]},
        (0, 1): {'head1': [30, 40], 'head2': [15]},
        (1, 0): {'head1': [100, 200], 'head2': [50]},
    }
    selected = ['row_000000_col_000000', 'row_000001_col_000000']

    result = mat_stats.count_label(raw_counts, selected)

    # head1: [10 + 30, 20 + 40] = [40, 60]
    # head2: [5 + 15] = [20]
    assert result == {'head1': [40, 60], 'head2': [20]}


def test_count_label_invalid_coordinate():
    '''
    Given: Block ID string with invalid coordinate format.
    When: Running count_label.
    Then: Raise ValueError.
    '''
    raw_counts = {(0, 0): {'head1': [10]}}
    with pytest.raises(ValueError, match='Invalid block coord string'):
        mat_stats.count_label(raw_counts, ['invalid_name.npz'])


# ----- `aggregate_image_stats` tests
def test_aggregate_image_stats_success(mocker):
    '''
    Given: A list of DataBlock paths with mock band metrics.
    When: Running aggregate_image_stats.
    Then: Correctly aggregate band means and standard deviations.
    '''
    mock_block = mocker.Mock()
    mock_block.data.image = numpy.random.rand(2, 64, 64).astype(numpy.float32)
    mock_block.manifest = {
        'image_stats': {
            'band_0': {
                'count': 100,
                'mean': 10.0,
                'm2': 400.0,
            },
            'band_1': {
                'count': 100,
                'mean': 20.0,
                'm2': 900.0,
            },
        }
    }

    mocker.patch('landseg.geopipe.core.DataBlock.load', return_value=mock_block)

    input_blocks = {pathlib.Path('block1.npz'), pathlib.Path('block2.npz')}
    result = mat_stats.aggregate_image_stats(input_blocks)

    assert len(result) == 2
    assert 'band_0' in result
    assert 'band_1' in result
    assert result['band_0']['total_count'] == 200
    assert result['band_0']['current_mean'] == pytest.approx(10.0)
    assert result['band_0']['std'] > 0.0


def test_aggregate_image_stats_channel_indices(mocker):
    '''
    Given: Input blocks and explicit channel index selection.
    When: Running aggregate_image_stats with channel_indices.
    Then: Aggregate statistics only for selected channels remapped.
    '''
    mock_block = mocker.Mock()
    mock_block.data.image = numpy.random.rand(3, 64, 64).astype(numpy.float32)
    mock_block.manifest = {
        'image_stats': {
            'band_0': {'count': 100, 'mean': 10.0, 'm2': 400.0},
            'band_1': {'count': 100, 'mean': 20.0, 'm2': 900.0},
            'band_2': {'count': 100, 'mean': 30.0, 'm2': 1600.0},
        }
    }

    mocker.patch('landseg.geopipe.core.DataBlock.load', return_value=mock_block)

    input_blocks = {pathlib.Path('block1.npz')}
    result = mat_stats.aggregate_image_stats(
        input_blocks, channel_indices=[2, 0]
    )

    assert len(result) == 2
    # band_0 maps to orig band_2 (mean 30.0)
    assert result['band_0']['current_mean'] == pytest.approx(30.0)
    # band_1 maps to orig band_0 (mean 10.0)
    assert result['band_1']['current_mean'] == pytest.approx(10.0)


def test_aggregate_image_stats_empty():
    '''
    Given: An empty input block set.
    When: Running aggregate_image_stats.
    Then: Raise ValueError.
    '''
    with pytest.raises(ValueError, match='input_blocks cannot be empty'):
        mat_stats.aggregate_image_stats(set())


def test_aggregate_image_stats_empty_channel_indices(mocker):
    '''
    Given: An empty list of channel indices.
    When: Running aggregate_image_stats.
    Then: Raise ValueError.
    '''
    mock_block = mocker.Mock()
    mock_block.data.image = numpy.random.rand(2, 64, 64).astype(numpy.float32)
    mocker.patch('landseg.geopipe.core.DataBlock.load', return_value=mock_block)

    with pytest.raises(ValueError, match='channel_indices cannot be empty'):
        mat_stats.aggregate_image_stats(
            {pathlib.Path('block1.npz')}, channel_indices=[]
        )


def test_aggregate_image_stats_invalid_channel_index_type(mocker):
    '''
    Given: Non-integer channel index.
    When: Running aggregate_image_stats.
    Then: Raise ValidationError.
    '''
    mock_block = mocker.Mock()
    mock_block.data.image = numpy.random.rand(2, 64, 64).astype(numpy.float32)
    mocker.patch('landseg.geopipe.core.DataBlock.load', return_value=mock_block)

    with pytest.raises(val.ValidationError):
        mat_stats.aggregate_image_stats(
            {pathlib.Path('block1.npz')},
            channel_indices=[True]  # type: ignore[list-item]
        )

    with pytest.raises(val.ValidationError, match='Expected an integer'):
        mat_stats.aggregate_image_stats(
            {pathlib.Path('block1.npz')},
            channel_indices=['0']  # type: ignore[list-item]
        )


def test_aggregate_image_stats_channel_index_out_of_bounds(mocker):
    '''
    Given: Out-of-bounds channel index.
    When: Running aggregate_image_stats.
    Then: Raise ValidationError.
    '''
    mock_block = mocker.Mock()
    mock_block.data.image = numpy.random.rand(2, 64, 64).astype(numpy.float32)
    mocker.patch('landseg.geopipe.core.DataBlock.load', return_value=mock_block)

    with pytest.raises(val.ValidationError, match='> maximum 1'):
        mat_stats.aggregate_image_stats(
            {pathlib.Path('block1.npz')}, channel_indices=[2]
        )

    with pytest.raises(val.ValidationError, match='< minimum 0'):
        mat_stats.aggregate_image_stats(
            {pathlib.Path('block1.npz')}, channel_indices=[-1]
        )


def test_aggregate_image_stats_missing_channel_in_manifest(mocker):
    '''
    Given: Block manifest missing a required channel entry.
    When: Running aggregate_image_stats.
    Then: Raise descriptive ValueError.
    '''
    mock_block = mocker.Mock()
    mock_block.data.image = numpy.random.rand(2, 64, 64).astype(numpy.float32)
    mock_block.manifest = {
        'image_stats': {
            'band_0': {'count': 100, 'mean': 10.0, 'm2': 400.0},
            # band_1 is missing
        }
    }
    mocker.patch('landseg.geopipe.core.DataBlock.load', return_value=mock_block)

    with pytest.raises(
        ValueError, match='Missing channel statistics for "band_1"'
    ):
        mat_stats.aggregate_image_stats({pathlib.Path('block1.npz')})


def test_aggregate_image_stats_missing_image_stats_key(mocker):
    '''
    Given: Block manifest lacking image_stats mapping.
    When: Running aggregate_image_stats.
    Then: Raise ValueError.
    '''
    mock_block = mocker.Mock()
    mock_block.data.image = numpy.random.rand(2, 64, 64).astype(numpy.float32)
    mock_block.manifest = {}
    mocker.patch('landseg.geopipe.core.DataBlock.load', return_value=mock_block)

    with pytest.raises(
        ValueError, match='is missing "image_stats" in manifest'
    ):
        mat_stats.aggregate_image_stats({pathlib.Path('block1.npz')})


def test_aggregate_image_stats_non_dict_image_stats(mocker):
    '''
    Given: Block manifest with non-dictionary image_stats.
    When: Running aggregate_image_stats.
    Then: Raise TypeError.
    '''
    mock_block = mocker.Mock()
    mock_block.data.image = numpy.random.rand(2, 64, 64).astype(numpy.float32)
    mock_block.manifest = {'image_stats': [1, 2, 3]}
    mocker.patch('landseg.geopipe.core.DataBlock.load', return_value=mock_block)

    with pytest.raises(TypeError, match='Expected a mapping'):
        mat_stats.aggregate_image_stats({pathlib.Path('block1.npz')})


def test_aggregate_image_stats_empty_summary(mocker):
    '''
    Given: Blocks with count=0 empty summaries.
    When: Running aggregate_image_stats.
    Then: Avoid division by zero and return default std.
    '''
    mock_block = mocker.Mock()
    mock_block.data.image = numpy.random.rand(1, 64, 64).astype(numpy.float32)
    mock_block.manifest = {
        'image_stats': {
            'band_0': {'count': 0, 'mean': 0.0, 'm2': 0.0},
        }
    }
    mocker.patch('landseg.geopipe.core.DataBlock.load', return_value=mock_block)

    result = mat_stats.aggregate_image_stats({pathlib.Path('block1.npz')})
    assert result['band_0']['total_count'] == 0
    assert result['band_0']['current_mean'] == 0.0
    assert result['band_0']['accum_m2'] == 0.0
    assert result['band_0']['std'] == 1.0


# ----- `_welfords_online` tests
def test_welfords_online_empty_input_preserves_aggregate():
    '''
    Given: Empty summary with count=0 and existing non-empty aggregate.
    When: Running _welfords_online.
    Then: Return aggregate unchanged without modification.
    '''
    current: contracts.ImageBandStats = {
        'total_count': 50,
        'current_mean': 12.5,
        'accum_m2': 200.0,
        'std': 0.0,
    }
    empty_summary = {'count': 0, 'mean': 0.0, 'm2': 0.0}

    result = mat_stats._welfords_online(empty_summary, current)

    assert result['total_count'] == 50
    assert result['current_mean'] == 12.5
    assert result['accum_m2'] == 200.0


def test_welfords_online_both_empty_no_zero_division():
    '''
    Given: Both aggregate and input summary having count=0.
    When: Running _welfords_online.
    Then: Safely return aggregate unchanged without ZeroDivisionError.
    '''
    current: contracts.ImageBandStats = {
        'total_count': 0,
        'current_mean': 0.0,
        'accum_m2': 0.0,
        'std': 0.0,
    }
    empty_summary = {'count': 0, 'mean': 0.0, 'm2': 0.0}

    result = mat_stats._welfords_online(empty_summary, current)

    assert result['total_count'] == 0
    assert result['current_mean'] == 0.0
    assert result['accum_m2'] == 0.0


def test_welfords_online_rejects_negative_count():
    '''
    Given: Summary with negative count.
    When: Running _welfords_online.
    Then: Raise ValidationError.
    '''
    current = _create_initial_stats()
    with pytest.raises(val.ValidationError, match='< minimum 0'):
        mat_stats._welfords_online(
            {'count': -1, 'mean': 10.0, 'm2': 50.0}, current
        )


def test_welfords_online_rejects_non_integer_count():
    '''
    Given: Summary with non-integer or boolean count.
    When: Running _welfords_online.
    Then: Raise ValidationError.
    '''
    current = _create_initial_stats()
    with pytest.raises(val.ValidationError):
        mat_stats._welfords_online(
            {'count': True, 'mean': 10.0, 'm2': 50.0}, current
        )
    with pytest.raises(val.ValidationError, match='Expected an integer'):
        mat_stats._welfords_online(
            {'count': '10', 'mean': 10.0, 'm2': 50.0}, current
        )


def test_welfords_online_rejects_non_finite_mean():
    '''
    Given: Summary with NaN or Inf mean.
    When: Running _welfords_online.
    Then: Raise ValidationError.
    '''
    current = _create_initial_stats()
    with pytest.raises(val.ValidationError, match='Expected a finite number'):
        mat_stats._welfords_online(
            {'count': 10, 'mean': float('nan'), 'm2': 50.0}, current
        )
    with pytest.raises(val.ValidationError, match='Expected a finite number'):
        mat_stats._welfords_online(
            {'count': 10, 'mean': float('inf'), 'm2': 50.0}, current
        )


def test_welfords_online_rejects_negative_m2():
    '''
    Given: Summary with negative m2.
    When: Running _welfords_online.
    Then: Raise ValidationError.
    '''
    current = _create_initial_stats()
    with pytest.raises(val.ValidationError, match='< minimum 0.0'):
        mat_stats._welfords_online(
            {'count': 10, 'mean': 5.0, 'm2': -1.0}, current
        )


def test_welfords_online_rejects_non_finite_m2():
    '''
    Given: Summary with NaN or Inf m2.
    When: Running _welfords_online.
    Then: Raise ValidationError.
    '''
    current = _create_initial_stats()
    with pytest.raises(val.ValidationError, match='Expected a finite number'):
        mat_stats._welfords_online(
            {'count': 10, 'mean': 5.0, 'm2': float('nan')}, current
        )
    with pytest.raises(val.ValidationError, match='Expected a finite number'):
        mat_stats._welfords_online(
            {'count': 10, 'mean': 5.0, 'm2': float('inf')}, current
        )


def test_welfords_online_rejects_missing_keys():
    '''
    Given: Summary missing required keys.
    When: Running _welfords_online.
    Then: Raise ValueError.
    '''
    current = _create_initial_stats()
    with pytest.raises(ValueError, match='Missing required stats key: "m2"'):
        mat_stats._welfords_online({'count': 10, 'mean': 5.0}, current)


def test_welfords_online_rejects_non_dict_summary():
    '''
    Given: Non-dict summary input.
    When: Running _welfords_online.
    Then: Raise TypeError.
    '''
    current = _create_initial_stats()
    bad_summary = [10, 5.0, 1.0]
    with pytest.raises(TypeError, match='Expected a mapping'):
        mat_stats._welfords_online(
            bad_summary, current  # type: ignore[arg-type]
        )


def test_welfords_online_rejects_corrupted_current_results():
    '''
    Given: Accumulated results with negative count or negative M2.
    When: Running _welfords_online.
    Then: Raise ValidationError.
    '''
    valid_summary = {'count': 10, 'mean': 5.0, 'm2': 10.0}

    bad_count: contracts.ImageBandStats = {
        'total_count': -1,
        'current_mean': 0.0,
        'accum_m2': 0.0,
        'std': 0.0,
    }
    with pytest.raises(val.ValidationError, match='< minimum 0'):
        mat_stats._welfords_online(valid_summary, bad_count)

    bad_m2: contracts.ImageBandStats = {
        'total_count': 10,
        'current_mean': 0.0,
        'accum_m2': -5.0,
        'std': 0.0,
    }
    with pytest.raises(val.ValidationError, match='< minimum 0.0'):
        mat_stats._welfords_online(valid_summary, bad_m2)


# ----- private helpers
def _create_initial_stats() -> contracts.ImageBandStats:
    '''Create a clean initial ImageBandStats dictionary.'''
    return {
        'total_count': 0,
        'current_mean': 0.0,
        'accum_m2': 0.0,
        'std': 0.0,
    }
