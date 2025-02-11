import sqlite3

# Configuration
DB_PATH = "iris_database.db"  # Database file name
TABLE_NAME = "iris_data"  # Table name


def create_database():
    try:
        # Connect to SQLite (creates the file if it doesn't exist)
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()

        # Create the iris_data table
        cursor.execute(
            f"""
        CREATE TABLE IF NOT EXISTS {TABLE_NAME} (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            sepal_length REAL NOT NULL,
            sepal_width REAL NOT NULL,
            petal_length REAL NOT NULL,
            petal_width REAL NOT NULL,
            target TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        """
        )

        # Commit and close
        conn.commit()
        print(f"Database '{DB_PATH}' and table '{TABLE_NAME}' created successfully.")

    except Exception as e:
        print(f"Error: {e}")

    finally:
        conn.close()


# Run the script
create_database()
