python src/setup.py develop
pip3 --cert /etc/ssl/certs/ca-certificates.crt install -r requirements.txt
pip3 --cert /etc/ssl/certs/ca-certificates.crt install -r src/api/requirements.txt
pip3 --cert /etc/ssl/certs/ca-certificates.crt install -r src/app_raspi/requirements.txt
pip3 --cert /etc/ssl/certs/ca-certificates.crt install -r src/predictor/requirements.txt

curl https://raw.githubusercontent.com/git/git/master/contrib/completion/git-completion.bash -o ~/.git-completion.bash --insecure
str="if [ -f ~/.git-completion.bash ] ; then . ~/.git-completion.bash ; fi"
echo $str >> ~/.bashrc
echo "Done"
