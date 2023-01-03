# See here for image contents: https://github.com/microsoft/vscode-dev-containers/tree/v0.245.2/containers/python-3/.devcontainer/base.Dockerfile

# [Choice] Python version (use -bullseye variants on local arm64/Apple Silicon): 3, 3.10, 3.9, 3.8, 3.7, 3.6, 3-bullseye, 3.10-bullseye, 3.9-bullseye, 3.8-bullseye, 3.7-bullseye, 3.6-bullseye, 3-buster, 3.10-buster, 3.9-buster, 3.8-buster, 3.7-buster, 3.6-buster
FROM mcr.microsoft.com/vscode/devcontainers/python:3.9@sha256:29ca94ddf0f254cb36e311ae59520c4dee33a35de7c0b2f1bd7b91650d368ee0

COPY .devcontainer/sgre_zscaler_cert.crt /usr/local/share/ca-certificates/
RUN cat /etc/ssl/certs/ca-certificates.crt /usr/local/share/ca-certificates/sgre_zscaler_cert.crt > /etc/ssl/certs/ca-certificates.crt
RUN apt-get update && apt-get install -y ca-certificates && \
    update-ca-certificates --fresh 
ENV NODE_EXTRA_CA_CERTS="/etc/ssl/certs/ca-certificates.crt"
