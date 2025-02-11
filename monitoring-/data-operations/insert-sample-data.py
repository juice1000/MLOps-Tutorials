import sqlite3

import pandas as pd

DB_PATH = "iris_database.db"
TABLE_NAME = "iris_data"


def insert_sample_data():
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()

        # Sample iris dataset rows
        sample_data = [
            (5.1, 3.5, 1.4, 0.2, "setosa"),
            (6.7, 3.0, 5.2, 2.3, "virginica"),
            (5.8, 2.7, 4.1, 1.0, "versicolor"),
        ]
        # Load iris dataset from CSV and insert into the table
        pd.read_csv("data/iris_new.csv").to_sql(
            TABLE_NAME, conn, if_exists="append", index=False
        )
        # Insert data into the table
        # cursor.executemany(
        #     f"""
        # INSERT INTO {TABLE_NAME} (sepal_length, sepal_width, petal_length, petal_width, target)
        # VALUES (?, ?, ?, ?, ?);
        # """,
        #     sample_data,
        # )

        conn.commit()
        print("Sample data inserted successfully.")

    except Exception as e:
        print(f"Error: {e}")

    finally:
        conn.close()


# Run the script
insert_sample_data()
