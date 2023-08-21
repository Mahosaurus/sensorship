#######################################
# Check of being in the right folder
#######################################
# Determine execution folder
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
# List folders in $SCRIPT_DIR
FOLDERS=$(ls -d "$SCRIPT_DIR"/*/)
# Assert that "app_raspi", "api", "src", and "tests" occur in "folders"
required_folders=("app_raspi" "api" "src" "tests")
# Check that level should contain api, app_raspi, src
for folder in "${required_folders[@]}"; do
  if [[ ! "${FOLDERS}" =~ ${folder} ]]; then
    echo "Folder '$folder' does not exist."
    exit 1
  fi
done

#######################################
# Pyenv
#######################################
# Prerequisites for pyenv
sudo apt update
sudo apt -y install make build-essential libssl-dev zlib1g-dev \
libbz2-dev libreadline-dev libsqlite3-dev wget curl llvm \
libncursesw5-dev xz-utils tk-dev libxml2-dev libxmlsec1-dev libffi-dev liblzma-dev

# Pyenv installation according to https://realpython.com/intro-to-pyenv/
curl -s -S -L https://raw.githubusercontent.com/pyenv/pyenv-installer/master/bin/pyenv-installer | bash
export PYENV_ROOT="$HOME/.pyenv"
command -v pyenv >/dev/null || export PATH="$PYENV_ROOT/bin:$PATH"
eval "$(pyenv init -)"
# Adding prev to bashrc
# https://unix.stackexchange.com/questions/21598/how-do-i-set-a-user-environment-variable-permanently-not-session
echo 'PYENV_ROOT="$HOME/.pyenv"' >> ${HOME}/.bash_profile
echo 'command -v pyenv >/dev/null || export PATH="$PYENV_ROOT/bin:$PATH"' >> ${HOME}/.bash_profile
echo 'eval "$(pyenv init -)"' >> ${HOME}/.bash_profile

# Check successful installation
pyenv --version

# Install python version
pyenv install -v 3.9.15
pyenv global 3.9.15

#######################################
# Install venv
#######################################
python -m venv venv
# https://pythonspeed.com/articles/activate-virtualenv-dockerfile/
export VIRTUAL_ENV=./venv
export PATH="$VIRTUAL_ENV/bin:$PATH"

#######################################
# Adding misc. configurations
#######################################
# Adding prev to bashrc
# https://unix.stackexchange.com/questions/21598/how-do-i-set-a-user-environment-variable-permanently-not-session
echo "export VIRTUAL_ENV=./venv" >> ${HOME}/.bash_profile
echo 'export PATH="$VIRTUAL_ENV/bin:$PATH"' >> ${HOME}/.bash_profile
echo 'export LINK=https://sensorndf.azurewebsites.net/sensor-data' >> ${HOME}/.bash_profile
# For Local Testing: echo 'export LINK=http://127.0.0.1:5000/sensor-data' >> ${HOME}/.bash_profile

# Adding bash profile to bashrc
# https://askubuntu.com/questions/121073/why-bash-profile-is-not-getting-sourced-when-opening-a-terminal
echo ". ~/.bash_profile" >> ${HOME}/.bashrc
python src/setup.py develop
pip install --no-cache-dir -r app_raspi/requirements.txt

# Add sensor start to cronjob
prog_sensor_path="$PWD"/run_sensor.sh
crontab -l > file; echo "@reboot sudo bash $prog_sensor_path" >> file; crontab file; rm file

#######################################
# Adding sensor configuration helpers
#######################################
