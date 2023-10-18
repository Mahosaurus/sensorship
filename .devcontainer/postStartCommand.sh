sudo update-ca-certificates

python src/setup.py develop
pip3 install -r requirements.txt --trusted-host pypi.org
pip3 install -r api/requirements.txt --trusted-host pypi.org
pip3 install -r app_raspi/requirements.txt --trusted-host pypi.org
pip3 install -r ml/requirements.txt --trusted-host pypi.org --trusted-host download.pytorch.org
