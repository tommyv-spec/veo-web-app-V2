# -*- coding: utf-8 -*-
"""
Database Migration: Add Flow Backend Fields

This migration adds fields needed for Flow backend support:

Jobs table:
- backend: 'api' or 'flow'
- flow_project_url: URL of Flow project
- flow_state_json: JSON state for Flow automation
- flow_needs_auth: Boolean flag if auth is needed

Clips table:
- flow_clip_id: ID from Flow UI
- output_url: S3/R2 URL for output video

Run this migration before deploying Flow backend.
"""

import os
import sys

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def run_migration_sqlite(db_path: str = None):
    """Run migration for SQLite database"""
    import sqlite3
    
    if db_path is None:
        from config import app_config
        db_path = str(app_config.data_dir / 'jobs.db')
    
    print(f"[Migration] SQLite: {db_path}")
    
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Jobs table migrations
    jobs_columns = [
        ("backend", "TEXT DEFAULT 'api'"),
        ("flow_project_url", "TEXT"),
        ("flow_state_json", "TEXT"),
        ("flow_needs_auth", "INTEGER DEFAULT 0"),
    ]
    
    for column_name, column_def in jobs_columns:
        try:
            cursor.execute(f"ALTER TABLE jobs ADD COLUMN {column_name} {column_def}")
            print(f"  ✓ Added jobs.{column_name}")
        except sqlite3.OperationalError as e:
            if "duplicate column name" in str(e).lower():
                print(f"  - jobs.{column_name} already exists")
            else:
                print(f"  ✗ Error adding jobs.{column_name}: {e}")
    
    # Clips table migrations
    clips_columns = [
        ("flow_clip_id", "TEXT"),
        ("output_url", "TEXT"),
    ]
    
    for column_name, column_def in clips_columns:
        try:
            cursor.execute(f"ALTER TABLE clips ADD COLUMN {column_name} {column_def}")
            print(f"  ✓ Added clips.{column_name}")
        except sqlite3.OperationalError as e:
            if "duplicate column name" in str(e).lower():
                print(f"  - clips.{column_name} already exists")
            else:
                print(f"  ✗ Error adding clips.{column_name}: {e}")
    
    conn.commit()
    conn.close()
    
    print("[Migration] SQLite migration complete")


def run_migration_postgresql(database_url: str = None):
    """Run migration for PostgreSQL database"""
    from sqlalchemy import create_engine, text
    
    if database_url is None:
        database_url = os.environ.get("DATABASE_URL")
        if database_url and database_url.startswith("postgres://"):
            database_url = database_url.replace("postgres://", "postgresql://", 1)
    
    if not database_url:
        print("[Migration] ERROR: No DATABASE_URL set")
        return
    
    print(f"[Migration] PostgreSQL")
    
    engine = create_engine(database_url)
    
    # Jobs table migrations
    jobs_migrations = [
        "ALTER TABLE jobs ADD COLUMN IF NOT EXISTS backend TEXT DEFAULT 'api'",
        "ALTER TABLE jobs ADD COLUMN IF NOT EXISTS flow_project_url TEXT",
        "ALTER TABLE jobs ADD COLUMN IF NOT EXISTS flow_state_json TEXT",
        "ALTER TABLE jobs ADD COLUMN IF NOT EXISTS flow_needs_auth BOOLEAN DEFAULT FALSE",
    ]
    
    # Clips table migrations
    clips_migrations = [
        "ALTER TABLE clips ADD COLUMN IF NOT EXISTS flow_clip_id TEXT",
        "ALTER TABLE clips ADD COLUMN IF NOT EXISTS output_url TEXT",
    ]
    
    with engine.connect() as conn:
        for sql in jobs_migrations + clips_migrations:
            try:
                conn.execute(text(sql))
                conn.commit()
                
                # Extract column name from SQL
                if "ADD COLUMN" in sql:
                    parts = sql.split("ADD COLUMN")
                    if len(parts) > 1:
                        col_part = parts[1].strip().split()[0]
                        if "IF NOT EXISTS" in sql:
                            col_part = parts[1].strip().split()[3]
                        print(f"  ✓ {col_part}")
            except Exception as e:
                print(f"  ✗ Error: {e}")
    
    print("[Migration] PostgreSQL migration complete")


def run_migration():
    """Run migration based on environment"""
    database_url = os.environ.get("DATABASE_URL")
    
    if database_url and "postgresql" in database_url.lower():
        run_migration_postgresql(database_url)
    elif database_url and "postgres" in database_url.lower():
        run_migration_postgresql(database_url)
    else:
        run_migration_sqlite()


if __name__ == "__main__":
    print("=" * 50)
    print("FLOW BACKEND DATABASE MIGRATION")
    print("=" * 50)
    print("")
    
    run_migration()
    
    print("")
    print("=" * 50)
    print("Migration complete!")
    print("=" * 50)
