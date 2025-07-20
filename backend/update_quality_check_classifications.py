#!/usr/bin/env python3
"""
Script to update WeeklyQualityCheck table with classification results from CSV file.

This script reads classification results from comment_classification_result_corrected.csv
and updates the database with:
- data_quality_check field (from classification column)  
- action field (from action_category column)

Usage:
    python update_quality_check_classifications.py --interim <interim_name>
    
Example:
    python update_quality_check_classifications.py --interim Interim1
"""

import asyncio
import argparse
import pandas as pd
import sys
from pathlib import Path
from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession

# Add backend to path to import modules
sys.path.append(str(Path(__file__).parent / "backend"))

from backend.core.db_connect import AsyncSessionLocal
from backend.models.monitor import Monitor  
from backend.models.WeeklyQualityCheck import WeeklyQualityCheck


async def get_monitor_id_by_name(session: AsyncSession, monitor_name: str) -> int:
    """Get monitor database ID by monitor business ID (e.g., 'FM01')"""
    result = await session.execute(
        select(Monitor.id).where(Monitor.monitor_id == monitor_name)
    )
    monitor_id = result.scalar_one_or_none()
    if monitor_id is None:
        raise ValueError(f"Monitor with monitor_id '{monitor_name}' not found")
    return monitor_id


async def update_or_create_quality_check(
    session: AsyncSession, 
    monitor_id: int, 
    interim: str, 
    classification: str, 
    action_category: str
) -> bool:
    """Update existing or create new WeeklyQualityCheck record"""
    
    # Check if record already exists
    existing_record = await session.execute(
        select(WeeklyQualityCheck).where(
            WeeklyQualityCheck.monitor_id == monitor_id,
            WeeklyQualityCheck.interim == interim
        )
    )
    existing = existing_record.scalar_one_or_none()
    
    if existing:
        # Get existing actions JSON and only update the "actions" key
        existing_actions = existing.actions or {}
        if isinstance(existing_actions, dict):
            # Update only the "actions" key, preserve all other keys
            updated_actions = existing_actions.copy()
            updated_actions["actions"] = action_category
        else:
            # If actions is not a dict, create new structure
            updated_actions = {"actions": action_category}
        
        # Update existing record - preserve other keys in actions JSON
        await session.execute(
            update(WeeklyQualityCheck)
            .where(
                WeeklyQualityCheck.monitor_id == monitor_id,
                WeeklyQualityCheck.interim == interim
            )
            .values(
                data_quality_check=classification,
                actions=updated_actions
            )
        )
        return False  # Updated existing
    else:
        # Create new record - start with just the actions key
        actions_json = {"actions": action_category}
        new_record = WeeklyQualityCheck(
            monitor_id=monitor_id,
            interim=interim,
            data_quality_check=classification,
            actions=actions_json
        )
        session.add(new_record)
        return True  # Created new


async def process_csv_file(csv_file_path: str, target_interim: str):
    """Process CSV file and update database"""
    
    # Read CSV file
    try:
        df = pd.read_csv(csv_file_path)
        print(f"✅ Successfully loaded CSV file: {csv_file_path}")
        print(f"📊 Total rows in CSV: {len(df)}")
    except Exception as e:
        print(f"❌ Error reading CSV file: {e}")
        return
    
    # Validate CSV columns
    required_columns = ['monitor_id', 'interim', 'classification', 'action_category']
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        print(f"❌ Missing required columns in CSV: {missing_columns}")
        return
    
    # Filter for target interim
    interim_df = df[df['interim'] == target_interim].copy()
    if interim_df.empty:
        print(f"❌ No records found for interim '{target_interim}' in CSV file")
        return
    
    print(f"🎯 Found {len(interim_df)} records for interim '{target_interim}'")
    
    # Process records
    async with AsyncSessionLocal() as session:
        created_count = 0
        updated_count = 0
        error_count = 0
        
        for index, row in interim_df.iterrows():
            monitor_name = row['monitor_id']
            classification = row['classification']
            action_category = row['action_category']
            
            try:
                # Get monitor ID
                monitor_id = await get_monitor_id_by_name(session, monitor_name)
                
                # Update or create record
                is_new = await update_or_create_quality_check(
                    session, monitor_id, target_interim, classification, action_category
                )
                
                if is_new:
                    created_count += 1
                    print(f"➕ Created new record for {monitor_name}")
                else:
                    updated_count += 1
                    print(f"🔄 Updated existing record for {monitor_name}")
                    
            except ValueError as e:
                print(f"⚠️  Warning: {e} (skipping {monitor_name})")
                error_count += 1
            except Exception as e:
                print(f"❌ Error processing {monitor_name}: {e}")
                error_count += 1
        
        # Commit all changes
        try:
            await session.commit()
            print(f"\n✅ Successfully committed all changes to database")
            print(f"📈 Summary:")
            print(f"   • Created new records: {created_count}")
            print(f"   • Updated existing records: {updated_count}")
            print(f"   • Errors/Skipped: {error_count}")
            print(f"   • Total processed: {created_count + updated_count}")
            
        except Exception as e:
            await session.rollback()
            print(f"❌ Error committing to database: {e}")
            print("🔄 All changes have been rolled back")


async def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Update WeeklyQualityCheck classifications from CSV')
    parser.add_argument(
        '--interim', 
        required=True, 
        help='Interim name to process (e.g., Interim1, Interim2, etc.)'
    )
    parser.add_argument(
        '--csv-file',
        default='comment_classification_result_corrected.csv',
        help='Path to CSV file (default: comment_classification_result_corrected.csv)'
    )
    
    args = parser.parse_args()
    
    # Validate CSV file exists
    csv_path = Path(args.csv_file)
    if not csv_path.exists():
        print(f"❌ CSV file not found: {csv_path}")
        print(f"📁 Current working directory: {Path.cwd()}")
        return
    
    print(f"🚀 Starting classification update process...")
    print(f"📂 CSV file: {csv_path}")
    print(f"🎯 Target interim: {args.interim}")
    print(f"{'='*50}")
    
    await process_csv_file(str(csv_path), args.interim)
    
    print(f"{'='*50}")
    print(f"🎉 Process completed!")


if __name__ == "__main__":
    asyncio.run(main()) 