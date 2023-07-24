import psycopg2


def send_data_to_postgres(host: str, dbname: str, user: str, password: str, data: str):
    """Send data to postgres database."""
    print(data)
    data = data["data"]
    date = data.split(",")[0]
    temperature = data.split(",")[1]
    humidity = data.split(",")[2]

    # Update connection string information
    sslmode = "require"
    # Construct connection string
    conn_string = "host={0} user={1} dbname={2} password={3} sslmode={4}".format(host, user, dbname, password, sslmode)

    conn = psycopg2.connect(conn_string)
    print("Connection established")

    cursor = conn.cursor()
    # Create a table
    cursor.execute("CREATE TABLE IF NOT EXISTS sensorship (id serial PRIMARY KEY, date VARCHAR(50), temperature NUMERIC(2), humidity NUMERIC(2));")
    print("Finished creating table")

    # Insert data into the table
    cursor.execute("INSERT INTO sensorship (date, temperature, humidity) VALUES (%s, %s, %s);", (date, temperature, humidity))
    print("Inserted 1 rows of data")

    conn.commit()

    cursor.close()
    conn.close()

def delete_database(host: str, dbname: str, user: str, password: str, data: str):
    # Update connection string information
    sslmode = "require"
    # Construct connection string
    conn_string = "host={0} user={1} dbname={2} password={3} sslmode={4}".format(host, user, dbname, password, sslmode)

    conn = psycopg2.connect(conn_string)
    print("Connection established")

    cursor = conn.cursor()
    # Create a table
    cursor.execute("DROP TABLE IF EXISTS sensorship")
    print("Finished killing table")

    conn.commit()

    cursor.close()
    conn.close()

def show_database(host: str, dbname: str, user: str, password: str):
    # Update connection string information
    sslmode = "require"
    # Construct connection string
    conn_string = "host={0} user={1} dbname={2} password={3} sslmode={4}".format(host, user, dbname, password, sslmode)

    conn = psycopg2.connect(conn_string)
    print("Connection established")

    cursor = conn.cursor()
    # Create a table
    cursor.execute("SELECT * FROM sensorship")
    rows = cursor.fetchall()
    print(rows)

    cursor.close()
    conn.close()

def fix_entries():
    pass #TODO: implementation

if __name__ == "__main__":
    show_database(host, dbname, user, password)
