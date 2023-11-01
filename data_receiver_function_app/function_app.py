import azure.functions as func

import logging
import os

import psycopg2

app = func.FunctionApp(http_auth_level=func.AuthLevel.ANONYMOUS)

@app.route(route="http_trigger", auth_level=func.AuthLevel.ANONYMOUS)
def http_trigger(req: func.HttpRequest) -> func.HttpResponse:
    # Some kind of JSON object is expected
    try:
        data = req.get_json()
    except ValueError:
        return func.HttpResponse(
             f"Received {req}, could not process it.",
             status_code=400
        )

    if data.get('secret_key', "") == os.environ.get("SECRET_KEY"):
        data = data["data"]
        validate_data(data)
        send_data_to_postgres(os.environ.get("POSTGRES_HOST"),
                              os.environ.get("POSTGRES_DBNAME"),
                              os.environ.get("POSTGRES_USER"),
                              os.environ.get("POSTGRES_PASSWORD"), data)
        return func.HttpResponse(
             f"Received and wrote {data} to database.",
             status_code=200
        )
    return func.HttpResponse(
            "Received a request which could not be processed.",
            status_code=400
    )

def validate_data(data: str) -> bool:
    pass

def send_data_to_postgres(host: str, dbname: str, user: str, password: str, data: str):
    """Send data to postgres database."""
    date = data.split(",")[0]
    temperature = data.split(",")[1]
    humidity = data.split(",")[2]

    # Update connection string information
    sslmode = "require"
    # Construct connection string
    conn_string = "host={0} user={1} dbname={2} password={3} sslmode={4}".format(host, user, dbname, password, sslmode)

    conn = psycopg2.connect(conn_string)
    logging.debug("Connection established")

    cursor = conn.cursor()

    # Create a table
    cursor.execute(f"""
                   CREATE TABLE IF NOT EXISTS {os.environ.get('POSTGRES_TABLE')}
                   (
                       id serial PRIMARY KEY,
                       date VARCHAR(50),
                       temperature NUMERIC(4, 2),
                       humidity NUMERIC(4, 2)
                    );"""
                )
    logging.debug("Finished creating table")

    # Insert data into the table
    cursor.execute(f"""
                   INSERT INTO {os.environ.get('POSTGRES_TABLE')}
                   (
                       date,
                       temperature,
                       humidity
                    ) VALUES ('{date}', {temperature}, {humidity})
                    ;"""
                )
    logging.debug("Inserted 1 rows of data")

    conn.commit()

    cursor.close()
    conn.close()
