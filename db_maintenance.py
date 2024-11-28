import sqlite3

def connect_to_db(db_name):
    """Connect to SQLite database."""
    try:
        conn = sqlite3.connect(db_name)
        print(f"Connected to database: {db_name}")
        return conn
    except sqlite3.Error as e:
        print(f"Error connecting to database: {e}")
        return None

def check_and_create_indexes(conn, table_name, indexes, composite_indexes):
    """
    Check for existing indexes and create them if they don't exist.

    :param conn: SQLite connection object.
    :param table_name: Name of the table to index.
    :param indexes: Dictionary with column names as keys and index names as values.
    :param composite_indexes: Dictionary with index names as keys and SQL CREATE statements as values.
    """
    cursor = conn.cursor()

    # Check and create single-column indexes
    for col, idx_name in indexes.items():
        try:
            cursor.execute(f"PRAGMA index_list('{table_name}')")
            existing_indexes = {row[1] for row in cursor.fetchall()}
            if idx_name not in existing_indexes:
                print(f"Creating index '{idx_name}' on column '{col}'")
                cursor.execute(f"CREATE INDEX {idx_name} ON {table_name}({col})")
            else:
                print(f"Index '{idx_name}' already exists.")
        except sqlite3.Error as e:
            print(f"Error checking/creating index {idx_name}: {e}")

    # Check and create composite indexes
    for idx_name, create_query in composite_indexes.items():
        try:
            cursor.execute(f"PRAGMA index_list('{table_name}')")
            existing_indexes = {row[1] for row in cursor.fetchall()}
            if idx_name not in existing_indexes:
                print(f"Creating composite index '{idx_name}'")
                cursor.execute(create_query)
            else:
                print(f"Composite index '{idx_name}' already exists.")
        except sqlite3.Error as e:
            print(f"Error checking/creating composite index {idx_name}: {e}")

def vacuum_database(conn):
    """Vacuum the SQLite database to optimize it."""
    try:
        print("Vacuuming the database...")
        conn.execute("VACUUM")
        print("Database vacuum completed.")
    except sqlite3.Error as e:
        print(f"Error during vacuum: {e}")

def main():
    db_name = "callsigns.db"
    table_name = "callsigns"
    indexes = {
        "zone": "idx_zone",
        "band": "idx_band",
        "timestamp": "idx_timestamp"
    }
    composite_indexes = {
        "idx_zone_band": f"CREATE INDEX idx_zone_band ON {table_name}(zone, band);",
        "idx_timestamp_zone": f"CREATE INDEX idx_timestamp_zone ON {table_name}(timestamp, zone);"
    }

    conn = connect_to_db(db_name)
    if conn:
        check_and_create_indexes(conn, table_name, indexes, composite_indexes)
        vacuum_database(conn)
        conn.close()

if __name__ == "__main__":
    main()
