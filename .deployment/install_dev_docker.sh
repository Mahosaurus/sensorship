sudo python src/setup.py develop
pip3 --cert /etc/ssl/certs/ca-certificates.crt install -r requirements.txt
pip3 --cert /etc/ssl/certs/ca-certificates.crt install -r api/requirements.txt
pip3 --cert /etc/ssl/certs/ca-certificates.crt install -r app_raspi/requirements.txt
pip3 --cert /etc/ssl/certs/ca-certificates.crt install -r ml/requirements.txt
pip3 install -r requirements.txt --trusted-host pypi.org
pip3 install -r api/requirements.txt --trusted-host pypi.org
pip3 install -r app_raspi/requirements.txt --trusted-host pypi.org
pip3 install -r ml/requirements.txt --trusted-host pypi.org --trusted-host download.pytorch.org

curl https://raw.githubusercontent.com/git/git/master/contrib/completion/git-completion.bash -o ~/.git-completion.bash --insecure
str="if [ -f ~/.git-completion.bash ] ; then . ~/.git-completion.bash ; fi"
echo $str >> ~/.bashrc
echo "Done"
