# Sensorship
## General description
- Project to monitor room conditions using a DHT22 sensor connected to a Rasperry Pi.

## How to deploy on Github
- Github Actions...

## How to start sensor
- Running the sender: `nohup bash run_sensor.sh &`

## How to retrain
- Projection based on LSTM models that incorporate the last 24h of data.

## Tips and tricks
### Env vars
#### Adding Env vars for app
- Secrets for the app to add in AppService in portal.azure.com (Settings -> Applications settings)
#### Finding out about secrets
- Secret to get from the same place or from Raspi or from SSH-Shell of Azure Service (env) and enter to bash script

### Data
- Cloud instance using Azure App Service to display values.
- Postgres
