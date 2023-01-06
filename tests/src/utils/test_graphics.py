import pytest
from data_visualization.graphics import PlotSensor

def test_map_time_to_time_of_day():
    # Test with valid input
    timestamp = "2022-12-16 00:00:00"
    expected_output = "Night"
    assert PlotSensor.map_time_to_time_of_day(timestamp) == expected_output

    timestamp = "2022-12-16 12:00:00"
    expected_output = "Day"
    assert PlotSensor.map_time_to_time_of_day(timestamp) == expected_output

    timestamp = "2022-12-16 18:00:00"
    expected_output = "Day"
    assert PlotSensor.map_time_to_time_of_day(timestamp) == expected_output

    # Test with invalid input
    with pytest.raises(TypeError):
        PlotSensor.map_time_to_time_of_day(1234567890)  # timestamp should be a string
    with pytest.raises(ValueError):
        PlotSensor.map_time_to_time_of_day("invalid_timestamp")  # invalid timestamp format
    with pytest.raises(ValueError):
        PlotSensor.map_time_to_time_of_day("2022-12-16 24:00:00")  # hour out of range

def test_determine_major_x_axis_interval():
    # Test with valid input
    timestamp = ["2022-01-01 00:00:00", "2022-12-31 23:59:59"]
    expected_output = 23
    assert PlotSensor.determine_major_x_axis_interval(timestamp) == expected_output

    timestamp = ["2022-12-15 00:00:00", "2022-12-31 23:59:59"]
    expected_output = 2
    assert PlotSensor.determine_major_x_axis_interval(timestamp) == expected_output

    timestamp = ["2022-01-01 00:00:00", "2022-03-31 23:59:59"]
    expected_output = 6
    assert PlotSensor.determine_major_x_axis_interval(timestamp) == expected_output

    # Test with invalid input
    with pytest.raises(TypeError):
        PlotSensor.determine_major_x_axis_interval(1234567890)  # timestamp should be a list of strings
    with pytest.raises(ValueError):
        PlotSensor.determine_major_x_axis_interval(["invalid_timestamp"])  # invalid timestamp format
    with pytest.raises(ValueError):
        PlotSensor.determine_major_x_axis_interval(["2022-01-01 00:00:00"])  # need at least two timestamps
