#!/usr/bin/env python3
"""
Script to import weekly quality check data from Excel file to database.
Usage: python import_weekly_quality_check.py
"""

import pandas as pd
import asyncio
import sys
import os
from datetime import datetime
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.future import select
from sqlalchemy import text

# Add backend to path
sys.path.append(os.path.dirname(__file__))

# Import all models to ensure they are registered with SQLAlchemy
from backend.models import Monitor, WeeklyQualityCheck
from backend.core.db_connect import get_session

# Configuration
EXCEL_FILE = "53100_Swansea Bay - Flow Survey Checksheet.xlsx"
SHEET_NAME = "Weekly quality check"

def get_interim_columns(interim_number: int) -> dict:
    """Calculate column positions for any interim number"""
    # Each interim takes 3 columns: silt, comment, action
    # Interim 1: columns 1,2,3 (B,C,D)
    # Interim 2: columns 4,5,6 (E,F,G)
    # Interim 3: columns 7,8,9 (H,I,J)
    # etc.
    base_col = (interim_number - 1) * 3 + 1
    return {
        "silt": base_col,
        "comment": base_col + 1,
        "action": base_col + 2
    }

async def get_monitor_by_id(db: AsyncSession, monitor_id: str) -> Monitor:
    """Get monitor by monitor_id"""
    try:
        result = await db.execute(
            select(Monitor).where(Monitor.monitor_id == monitor_id)
        )
        return result.scalars().first()
    except Exception as e:
        print(f"Error getting monitor {monitor_id}: {e}")
        return None

async def check_existing_record(db: AsyncSession, monitor_id: int, interim: str) -> WeeklyQualityCheck:
    """Get existing record for the given monitor and interim"""
    try:
        result = await db.execute(
            select(WeeklyQualityCheck).where(
                WeeklyQualityCheck.monitor_id == monitor_id,
                WeeklyQualityCheck.interim == interim
            )
        )
        return result.scalars().first()
    except Exception as e:
        print(f"Error checking existing record for monitor {monitor_id}, interim {interim}: {e}")
        return None

async def import_interim_data(interim_number: int, check_date: datetime.date):
    """Import data for a specific interim period"""
    
    # Read Excel file
    try:
        df = pd.read_excel(EXCEL_FILE, sheet_name=SHEET_NAME, header=None)
        print(f"Successfully loaded Excel file: {EXCEL_FILE}")
        print(f"Excel file shape: {df.shape}")
    except Exception as e:
        print(f"Error reading Excel file: {e}")
        return
    
    # Validate interim number
    if interim_number < 1:
        print(f"Invalid interim number: {interim_number}. Must be 1 or greater.")
        return
    
    columns = get_interim_columns(interim_number)
    print(f"Interim {interim_number} columns: {columns}")
    
    # Check if columns exist in the dataframe
    max_col = max(columns.values())
    if max_col >= len(df.columns):
        print(f"Warning: Interim {interim_number} columns extend beyond available columns in Excel file.")
        print(f"Available columns: {len(df.columns)}, Required columns: {max_col + 1}")
        return
    
    # Get database session
    async for db in get_session():
        try:
            updated_count = 0
            not_found_count = 0
            error_count = 0
            
            # Start from row 4 (index 3) where data begins
            for row_idx in range(3, len(df)):
                try:
                    # Get monitor ID from column A
                    monitor_id_raw = df.iloc[row_idx, 0]
                    monitor_id_str = str(monitor_id_raw).strip() if pd.notna(monitor_id_raw) else None
                    
                    # Skip empty rows or invalid monitor IDs
                    if not monitor_id_str or monitor_id_str == 'nan' or monitor_id_str == '':
                        continue
                    
                    # Get comment only
                    comment_value = df.iloc[row_idx, columns["comment"]]
                    comment = str(comment_value).strip() if pd.notna(comment_value) else None
                    
                    # Skip if no comment
                    if not comment or comment == '':
                        continue
                    
                    print(f"Processing: Monitor {monitor_id_str}, Comment: {comment[:50]}...")
                    
                    # Find monitor in database
                    monitor = await get_monitor_by_id(db, monitor_id_str)
                    if not monitor:
                        print(f"Warning: Monitor {monitor_id_str} not found in database, skipping...")
                        continue
                    
                    # Find existing record
                    existing_record = await check_existing_record(db, monitor.id, f"Interim{interim_number}")
                    if not existing_record:
                        print(f"Warning: No existing record found for {monitor_id_str}, Interim{interim_number}")
                        not_found_count += 1
                        continue
                    
                    # Update the existing record with comment
                    # Use the same structure as seen in the database: {"notes": "comment"}
                    existing_record.comments = {"notes": comment}
                    
                    updated_count += 1
                    print(f"Updated record for {monitor_id_str}, Interim{interim_number}")
                    
                except Exception as e:
                    error_count += 1
                    print(f"Error processing row {row_idx}: {e}")
                    continue
            
            # Commit all changes
            await db.commit()
            print(f"\nUpdate completed!")
            print(f"Updated: {updated_count} records")
            print(f"Not found: {not_found_count} records")
            print(f"Errors: {error_count} records")
            
        except Exception as e:
            await db.rollback()
            print(f"Error during update: {e}")
            import traceback
            traceback.print_exc()
            raise

def main():
    """Main function"""
    print("Weekly Quality Check Data Import Tool")
    print("=" * 50)
    
    # Check if Excel file exists
    if not os.path.exists(EXCEL_FILE):
        print(f"Error: Excel file '{EXCEL_FILE}' not found in current directory.")
        print(f"Current directory: {os.getcwd()}")
        return
    
    # Get user input for interim
    while True:
        try:
            interim = int(input("Enter interim number (1 or greater): "))
            if interim >= 1:
                break
            else:
                print("Please enter a number 1 or greater.")
        except ValueError:
            print("Please enter a valid number.")
    
    # Get user input for check date
    while True:
        try:
            date_str = input("Enter check date (YYYY-MM-DD): ")
            check_date = datetime.strptime(date_str, "%Y-%m-%d").date()
            break
        except ValueError:
            print("Please enter a valid date in YYYY-MM-DD format.")
    
    print(f"\nImporting data for Interim{interim} with check date {check_date}")
    print("Starting import...")
    
    # Run the import
    asyncio.run(import_interim_data(interim, check_date))

if __name__ == "__main__":
    main() 