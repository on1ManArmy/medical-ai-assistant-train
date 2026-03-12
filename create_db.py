import sqlite3
import os

DB_PATH = "./database/DeepCareX.db"

# Create database folder if not exists
os.makedirs("./database", exist_ok=True)

def create_database():

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    print("Creating tables...")

    # USER TABLE
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS USER(
        NAME TEXT NOT NULL,
        EMAIL TEXT NOT NULL,
        PASSWORD TEXT NOT NULL
    )
    """)

    # CONTACT TABLE
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS CONTACT(
        NAME TEXT,
        EMAIL TEXT,
        CONTACT TEXT,
        MESSAGE TEXT
    )
    """)

    # NEWSLETTER TABLE
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS NEWSLETTER(
        EMAIL TEXT
    )
    """)

    # PATIENTS TABLE
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS PATIENTS(
        NAME TEXT,
        EMAIL TEXT,
        ID TEXT,
        CONTACT TEXT,
        COUNTRY TEXT,
        STATE TEXT,
        PIN TEXT,
        GENDER TEXT,
        AGE TEXT,
        DISEASE TEXT,
        RESULT TEXT
    )
    """)

    conn.commit()
    conn.close()

    print("Database and tables created successfully!")

if __name__ == "__main__":
    create_database()