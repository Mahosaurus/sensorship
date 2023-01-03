python src/setup.py develop
pip3 --cert /etc/ssl/certs/ca-certificates.crt install -r requirements.txt
pip3 --cert /etc/ssl/certs/ca-certificates.crt install -r src/api/requirements.txt
pip3 --cert /etc/ssl/certs/ca-certificates.crt install -r src/app_raspi/requirements.txt
pip3 --cert /etc/ssl/certs/ca-certificates.crt install -r src/predictor/requirements.txt