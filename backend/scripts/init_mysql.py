import pymysql
from loguru import logger

def create_db():
    try:
        # Connect to MySQL (without specifying a database)
        conn = pymysql.connect(
            host='localhost',
            user='root',
            password=''
        )
        cursor = conn.cursor()
        
        # Create database
        db_name = "hawkeye_ai"
        cursor.execute(f"CREATE DATABASE IF NOT EXISTS {db_name}")
        logger.success(f"Database '{db_name}' created or already exists.")
        
        cursor.close()
        conn.close()
    except Exception as e:
        logger.error(f"Failed to create database: {e}")
        logger.info("Make sure XAMPP MySQL is running!")

if __name__ == "__main__":
    create_db()
