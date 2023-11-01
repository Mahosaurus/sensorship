import os

import pandas as pd
import psycopg2

from src.config import DATA_COLUMNS

class PostgresInteraction():
    def __init__(self, host: str, dbname: str, user: str, password: str, table: str=os.getenv("POSTGRES_TABLE")):
        self.host = host
        self.dbname = dbname
        self.user = user
        self.password = password
        self.table = table
        self.conn_string = "host={0} user={1} dbname={2} password={3} sslmode={4}".format(self.host, self.user, self.dbname, self.password, "require")
        self.conn: psycopg2.extensions.connection=None,
        self.cursor: psycopg2.extensions.cursor=None

    def open_cursor_and_conn(self):
        """Get cursor to postgres database."""
        print(self.conn_string )
        self.conn = psycopg2.connect(self.conn_string)
        print("Connection established")
        self.cursor = self.conn.cursor()

    def close_cursor_and_conn(self):
        """Close cursor to postgres database."""
        self.conn.commit()
        self.cursor.close()
        self.conn.close()

    def send_data_to_postgres(self, data: str):
        """Send data to postgres database."""
        data = data["data"]
        date = data.split(",")[0]
        temperature = data.split(",")[1]
        humidity = data.split(",")[2]

        self.open_cursor_and_conn()

        # Create a table
        self.cursor.execute(f"CREATE TABLE IF NOT EXISTS {self.table} (id serial PRIMARY KEY, date VARCHAR(50), temperature NUMERIC(4, 2), humidity NUMERIC(4, 2));")
        print("Finished creating table")

        # Insert data into the table
        self.cursor.execute(f"INSERT INTO {self.table} (date, temperature, humidity) VALUES (%s, %s, %s);", (date, temperature, humidity))
        print("Inserted 1 rows of data")

        self.close_cursor_and_conn()

    def delete_database(self):
        """Delete database."""
        self.open_cursor_and_conn()

        # Bobby tables
        self.cursor.execute(f"DROP TABLE IF EXISTS {self.table}")
        print("Finished killing table")

        self.close_cursor_and_conn()

    def load_data(self) -> pd.DataFrame:
        """Load data from database."""
        self.open_cursor_and_conn()

        # Create a table
        print("Loading data from Postgres")
        self.cursor.execute(f"SELECT * FROM {self.table}")
        rows = self.cursor.fetchall()
        print("Finished loading data from Postgres")

        self.close_cursor_and_conn()

        data = self.transform_data(rows)
        return data

    @staticmethod
    def transform_data(rows: list) -> pd.DataFrame:
        """Convert data to pandas dataframe."""
        # Convert to pandas dataframe
        data = pd.DataFrame(rows, columns=['id', 'timestamp', 'temperature', 'humidity'])
        data = data.sort_values(by=['timestamp'])
        # Restrict columns to timestamp, temperature, humidity
        data = data[DATA_COLUMNS]
        # Convert temperature and humidity to float
        data['temperature'] = data['temperature'].astype(float)
        data['humidity'] = data['humidity'].astype(float)
        return data
