SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
app_path="$SCRIPT_DIR"/src/app_raspi/app.py
source "$SCRIPT_DIR"/venv/bin/activate
python "$app_path"
