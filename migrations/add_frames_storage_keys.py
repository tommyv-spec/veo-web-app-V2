"""
Migration: Add frames_storage_keys column to jobs table

This column stores R2/S3 keys for uploaded frames, allowing redos to work
even if local ephemeral storage is cleared (common on Render).

Run: python -c "from migrations.add_frames_storage_keys import migrate; migrate()"
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sqlalchemy import text
from models import get_db


def migrate():
    """Add frames_storage_keys column to jobs table"""
    print("=" * 50)
    print("Migration: Add frames_storage_keys column")
    print("=" * 50)
    
    with get_db() as db:
        # Check if column exists
        try:
            result = db.execute(text("SELECT frames_storage_keys FROM jobs LIMIT 1"))
            print("✓ Column 'frames_storage_keys' already exists")
            return True
        except Exception:
            pass
        
        # Add column
        try:
            db.execute(text(
                "ALTER TABLE jobs ADD COLUMN frames_storage_keys TEXT"
            ))
            db.commit()
            print("✓ Added column 'frames_storage_keys' to jobs table")
            return True
        except Exception as e:
            print(f"✗ Failed to add column: {e}")
            return False


if __name__ == "__main__":
    migrate()
