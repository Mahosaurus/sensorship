# Install prereq for venv
sudo apt -y install python3.8-venv
# Install venv
python3 -m venv venv

# https://pythonspeed.com/articles/activate-virtualenv-dockerfile/
export VIRTUAL_ENV=./venv
export PATH="$VIRTUAL_ENV/bin:$PATH"

# Adding prev to bashrc
# https://unix.stackexchange.com/questions/21598/how-do-i-set-a-user-environment-variable-permanently-not-session
echo "export VIRTUAL_ENV=./venv" >> ${HOME}/.bash_profile
echo 'export PATH="$VIRTUAL_ENV/bin:$PATH"' >> ${HOME}/.bash_profile
#F or Local Testing
echo 'export LINK=http://127.0.0.1:5000/sensor-data' >> ${HOME}/.bash_profile

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
source "$SCRIPT_DIR"/venv/bin/activate
python src/setup.py develop
pip3 install -r requirements.txt
pip3 install -r src/api/requirements.txt

