#!/usr/bin/env python
"""Test script to verify the timestamp parsing fix"""

from pathlib import Path
import pandas as pd
import sys
sys.path.append('.')

# Import the fixed function
from backfill import ingest_discord

def test_timestamp_parsing():
    print("Testing Discord timestamp parsing fix...")
    
    # Test the ingestion
    df = ingest_discord(Path('data/discord'))
    
    print(f"✅ Loaded {len(df)} Discord messages")
    print(f"✅ Timestamp dtype: {df['timestamp'].dtype}")
    
    # Test that .dt accessor works
    try:
        sample_dates = df['timestamp'].dt.date.head()
        print(f"✅ .dt accessor works: {sample_dates.iloc[0]}")
    except Exception as e:
        print(f"❌ .dt accessor failed: {e}")
        return False
    
    # Test that date column exists and works
    try:
        date_range = f"{df['date'].min()} to {df['date'].max()}"
        print(f"✅ Date column works: {date_range}")
    except Exception as e:
        print(f"❌ Date column failed: {e}")
        return False
    
    # Test filtering by date
    try:
        from datetime import datetime
        test_date = datetime(2023, 6, 29).date()
        filtered = df[df['date'] == test_date]
        print(f"✅ Date filtering works: {len(filtered)} messages on {test_date}")
    except Exception as e:
        print(f"❌ Date filtering failed: {e}")
        return False
    
    print("\n🎉 All tests passed! The timestamp parsing fix is working correctly.")
    return True

if __name__ == "__main__":
    test_timestamp_parsing()
