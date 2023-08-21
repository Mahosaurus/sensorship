SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
app_path="$SCRIPT_DIR"/app_raspi/app.py
source "$SCRIPT_DIR"/venv/bin/activate
# This must be filled
export SECRET_KEY=""
# TODO: Check that SECRET_KEY is not empty
python "$app_path"
