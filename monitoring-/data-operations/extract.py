import sqlite3

import pandas as pd

# Configuration
DB_PATH = "iris_database.db"  # Path to your SQLite database
TABLE_NAME = "iris_data"  # Table to query
TIMESTAMP_COLUMN = "created_at"  # Column storing timestamps
EXPORT_FILE = "data/iris.csv"  # Output CSV file
# TIMESTAMP = "2024-02-01 00:00:00"  # Change this to your desired timestamp


# TODO: utilize the TIMESTAMP variable to extract data from the database


def extract_data_from_sqlite(timestamp):
    try:
        # Connect to SQLite database
        conn = sqlite3.connect(DB_PATH)

        # SQL query to extract new data
        query = f"""
        SELECT sepal_length, sepal_width, petal_length, petal_width, target FROM {TABLE_NAME}
        WHERE {TIMESTAMP_COLUMN} > ?
        """

        # Load data into a Pandas DataFrame
        df = pd.read_sql_query(query, conn, params=(timestamp,))

        # Save to CSV
        df.to_csv(EXPORT_FILE, index=False)

        print(f"Data extracted successfully and saved to {EXPORT_FILE}")

    except Exception as e:
        print(f"Error: {e}")

    finally:
        conn.close()


# Run the extraction
extract_data_from_sqlite()
