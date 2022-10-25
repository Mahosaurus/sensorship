# Prerequisites for pyenv
apt-get install -yy make build-essential libssl-dev zlib1g-dev \
libbz2-dev libreadline-dev libsqlite3-dev wget curl llvm libncurses5-dev \
libncursesw5-dev xz-utils tk-dev libffi-dev liblzma-dev python-openssl

# Pyenv installation according to https://realpython.com/intro-to-pyenv/
curl -s -S -L https://raw.githubusercontent.com/pyenv/pyenv-installer/master/bin/pyenv-installer | bash
export PYENV_ROOT="$HOME/.pyenv"
command -v pyenv >/dev/null || export PATH="$PYENV_ROOT/bin:$PATH"
eval "$(pyenv init -)"
pyenv --version

pyenv install -v 3.9.15
pyenv global 3.9.15

export VIRTUAL_ENV=./venv
export PATH="$VIRTUAL_ENV/bin:$PATH"
python -m venv venv
source ./venv/bin/activate
pip install --no-cache-dir -r requirements.txt