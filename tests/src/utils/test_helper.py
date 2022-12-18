import pytest
from src.utils.helpers import parse_data_points, compile_data_point

def test_parse_data_points():
    # Test with a simple input
    data = "2022-01-01T00:00:00Z,0.0,25.0,50.0\n2022-01-01T01:00:00Z,1.0,26.0,51.0\n"
    expected_timestamps = ["2022-01-01T00:00:00Z", "2022-01-01T01:00:00Z"]
    expected_temperatures = [25.0, 26.0]
    expected_rel_humidity = [50.0, 51.0]
    timestamps, temperatures, rel_humidity = parse_data_points(data)
    assert timestamps == expected_timestamps
    assert temperatures == expected_temperatures
    assert rel_humidity == expected_rel_humidity
    
    # Test with an empty input
    data = ""
    expected_timestamps = []
    expected_temperatures = []
    expected_rel_humidity = []
    timestamps, temperatures, rel_humidity = parse_data_points(data)
    assert timestamps == expected_timestamps
    assert temperatures == expected_temperatures
    assert rel_humidity == expected_rel_humidity
    
    # Test with a large input
    data = "\n".join(["2022-01-01T00:00:00Z,0.0,{},{}".format(i, i+50) for i in range(1000)]) + "\n"
    expected_timestamps = ["2022-01-01T00:00:00Z"] * 1000
    expected_temperatures = list(range(1000))
    expected_rel_humidity = [i+50 for i in range(1000)]
    timestamps, temperatures, rel_humidity = parse_data_points(data)
    assert timestamps == expected_timestamps
    assert temperatures == expected_temperatures
    assert rel_humidity == expected_rel_humidity

def test_compile_data_point():
    # Test with a simple input
    timestamp = 1623456789
    temperature = "25.0"
    humidity = "50.0"
    expected_output = "2021-06-12 00:13:09, TO_REMOVE, 25.0, 50.0\n"
    output = compile_data_point(timestamp, temperature, humidity)
    assert output == expected_output
    
    # Test with an empty input
    timestamp = 0
    temperature = ""
    humidity = ""
    expected_output = "1970-01-01 00:00:00, TO_REMOVE, , \n"
    output = compile_data_point(timestamp, temperature, humidity)
    print(output)
    assert output == expected_output
    
    # Test with a large input
    timestamp = 1623456789
    temperature = "1000000.0"
    humidity = "1000050.0"
    expected_output = "2021-06-12 00:13:09, TO_REMOVE, 1000000.0, 1000050.0\n"
    output = compile_data_point(timestamp, temperature, humidity)
    assert output == expected_output    

    # Test with invalid input
    with pytest.raises(TypeError):
        compile_data_point("invalid_timestamp", temperature, humidity)
    with pytest.raises(AssertionError):
        compile_data_point(timestamp, 12.3, humidity)
    with pytest.raises(AssertionError):
        compile_data_point(timestamp, temperature, True)    