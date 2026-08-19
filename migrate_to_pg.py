import os
import sqlite3
from sqlalchemy import create_engine, MetaData
from sqlalchemy.orm import sessionmaker
from dotenv import load_dotenv

# Load env variables
load_dotenv()

sqlite_url = "sqlite:///renalscan.db"
pg_url = os.environ.get("DATABASE_URL")

if not pg_url:
    print("ERROR: DATABASE_URL environment variable is not set. Cannot run migration.")
    exit(1)

if pg_url.startswith("postgres://"):
    pg_url = pg_url.replace("postgres://", "postgresql://", 1)

print("Starting migration from SQLite to PostgreSQL...")
print(f"Source: {sqlite_url}")
print(f"Destination: {pg_url}")

# Create Engines
src_engine = create_engine(sqlite_url)
dst_engine = create_engine(pg_url)

# Bind Session
src_Session = sessionmaker(bind=src_engine)
dst_Session = sessionmaker(bind=dst_engine)

src_session = src_Session()
dst_session = dst_Session()

# Reflect schemas
src_metadata = MetaData()
src_metadata.reflect(bind=src_engine)

dst_metadata = MetaData()
dst_metadata.reflect(bind=dst_engine)

# Order of tables to migrate to prevent foreign key constraint violations
tables_to_migrate = [
    'users',
    'patients',
    'encounters',
    'scans',
    'ai_results',
    'reviews',
    'reports'
]

# We disable foreign key checks or perform in correct order
conn = dst_engine.connect()

try:
    for table_name in tables_to_migrate:
        if table_name not in src_metadata.tables:
            print(f"Skipping table {table_name} (not found in source)")
            continue
            
        print(f"Migrating table: {table_name}...")
        
        # Read all rows from source SQLite table
        src_table = src_metadata.tables[table_name]
        rows = src_session.execute(src_table.select()).fetchall()
        
        if not rows:
            print(f"Table {table_name} is empty. Skipping.")
            continue
            
        dst_table = dst_metadata.tables[table_name]
        
        # Clear existing table content in destination to avoid duplicates
        dst_session.execute(dst_table.delete())
        
        # Map values and insert
        for row in rows:
            # Row is a RowMapping or tuple depending on SQLAlchemy version
            data = dict(row._mapping if hasattr(row, "_mapping") else row)
            dst_session.execute(dst_table.insert().values(**data))
            
        dst_session.commit()
        print(f"Successfully migrated {len(rows)} rows for {table_name}.")
        
        # Reset PostgreSQL serial sequences to prevent PK increment conflicts
        try:
            # PostgreSQL command to set sequence to max id or 1
            from sqlalchemy import text
            seq_reset_sql = text(f"SELECT setval(pg_get_serial_sequence('{table_name}', 'id'), COALESCE(MAX(id), 1)) FROM {table_name}")
            dst_session.execute(seq_reset_sql)
            dst_session.commit()
            print(f"Successfully reset serial sequence for table: {table_name}")
        except Exception as seq_err:
            print(f"Warning: Could not reset serial sequence for {table_name}: {seq_err}")
            dst_session.rollback()

    print("\nDATABASE MIGRATION COMPLETED SUCCESSFULLY!")
except Exception as e:
    dst_session.rollback()
    print(f"\nFATAL ERROR DURING MIGRATION: {e}")
    exit(1)
finally:
    src_session.close()
    dst_session.close()
