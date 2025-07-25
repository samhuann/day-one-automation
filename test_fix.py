#!/usr/bin/env python
from backfill import ingest_discord
from pathlib import Path

print('Testing Discord timestamp parsing...')
df_discord = ingest_discord(Path('data/discord'))
print(f'Discord messages loaded: {len(df_discord)}')
print(f'Timestamp dtype: {df_discord["timestamp"].dtype}')

if len(df_discord) > 0:
    print(f'Sample timestamps:')
    print(df_discord[['timestamp', 'date']].head())
    print(f'Date range: {df_discord["date"].min()} to {df_discord["date"].max()}')
    
    # Test the .dt accessor
    try:
        print(f'Testing .dt accessor: {df_discord["timestamp"].dt.date.head()}')
        print('✅ .dt accessor works!')
    except Exception as e:
        print(f'❌ .dt accessor failed: {e}')
