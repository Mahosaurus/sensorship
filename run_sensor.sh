SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
app_path="$SCRIPT_DIR"/app_raspi/app.py
source "$SCRIPT_DIR"/venv/bin/activate
# This must be filled
export SECRET_KEY=""
# If secret key is empty, exit
if [ -z "$SECRET_KEY" ]; then
    echo "SECRET_KEY is empty. Exiting..."
    exit 1
fi
# TODO: Check that SECRET_KEY is not empty
python "$app_path"
