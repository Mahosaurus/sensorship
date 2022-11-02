# Prerequisites for pyenv
apt-get install -yy make build-essential libssl-dev zlib1g-dev \
libbz2-dev libreadline-dev libsqlite3-dev wget curl llvm libncurses5-dev \
libncursesw5-dev xz-utils tk-dev libffi-dev liblzma-dev python-openssl

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

pyenv --version

# Install python version
pyenv install -v 3.9.15
pyenv global 3.9.15

# Install venv
python -m venv venv
# https://pythonspeed.com/articles/activate-virtualenv-dockerfile/
export VIRTUAL_ENV=./venv
export PATH="$VIRTUAL_ENV/bin:$PATH"
# Adding prev to bashrc
# https://unix.stackexchange.com/questions/21598/how-do-i-set-a-user-environment-variable-permanently-not-session
echo "export VIRTUAL_ENV=./venv" >> ${HOME}/.bash_profile
echo 'export PATH="$VIRTUAL_ENV/bin:$PATH"' >> ${HOME}/.bash_profile
echo 'export LINK=https://sensorndf.azurewebsites.net/sensor-data' >> ${HOME}/.bash_profile
# Local Testing: echo 'export LINK=http://127.0.0.1:5000/sensor-data' >> ${HOME}/.bash_profile
# https://askubuntu.com/questions/121073/why-bash-profile-is-not-getting-sourced-when-opening-a-terminal
echo ". ~/.bash_profile" >> ${HOME}/.bashrc
pip install --no-cache-dir -r requirements.txt
