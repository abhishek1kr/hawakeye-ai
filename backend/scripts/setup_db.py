import mysql.connector
from src.api.db import engine, Base

def setup_mysql():
    # Try to connect to MySQL to create the database first
    try:
        conn = mysql.connector.connect(
            host="localhost",
            user="root",
            password=""
        )
        cursor = conn.cursor()
        cursor.execute("CREATE DATABASE IF NOT EXISTS hawkeye_ai")
        print("Database 'hawkeye_ai' checked/created.")
        conn.close()
    except Exception as e:
        print(f"Error connecting to MySQL: {e}")
        return

    # Now create tables
    try:
        print("Creating tables...")
        Base.metadata.create_all(bind=engine)
        print("Tables created successfully.")
    except Exception as e:
        print(f"Error creating tables: {e}")

if __name__ == "__main__":
    setup_mysql()
