import datetime

def map_time_to_time_of_day(timestamp: str) -> str:
    """ Extract time of day from ts """
    hour = int(datetime.datetime.fromtimestamp(timestamp).strftime('%H'))
    class Switch(dict):
        """ Helper class to emulate switch statement """
        def __getitem__(self, item):
            for key in self.keys():                 # iterate over the intervals
                if item in key:                     # if the argument is in that interval
                    return super().__getitem__(key) # return its associated value
            raise KeyError(item)                    # if not in any interval, raise KeyError
    switch = Switch({
        range(0, 6): 'Night',
        range(6, 10): 'Morning',
        range(10, 14): 'Day',
        range(14, 17): 'Afternoon',
        range(17, 21): 'Evening',
        range(21, 24): 'Night'
    })
    time_of_day = switch[hour]
    return time_of_day

def format_timestamp(timestamp: str) -> str:
    """ Convert ts to human readable """
    return datetime.datetime.fromtimestamp(timestamp).strftime('%Y-%m-%d %H:%M:%S')

def compile_data_point(timestamp: str, temperature: str, humidity: str) -> str:
    """ Compiles the information for one data point for text file"""
    formatted_timestamp = format_timestamp(timestamp)
    time_of_day = map_time_to_time_of_day(timestamp)
    out_str = f"{formatted_timestamp}, {time_of_day}, {temperature}, {humidity}\n"
    return out_str
