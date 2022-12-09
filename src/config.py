import os

def get_repo_root() -> str:
    """
    Returns the absolute path to the root of the  repository
    specific to the system the code is run on.
    """
    path_to_this_file = os.path.dirname(os.path.realpath(__file__)) + "/"
    return path_to_this_file

API_DATA_PATH = os.path.join("/home", "site", "outcome_remote.txt")
APP_TEST_DATA_PATH = os.path.join(get_repo_root(), "outcome_local.txt")

DATA_COLUMNS = ["timestamp", "remove", "temperature", "humidity"]
DATA_SEP = ","

LSTM_INPUT_HISTORY = 24