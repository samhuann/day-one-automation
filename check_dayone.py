#!/usr/bin/env python
import sys
sys.path.append('.')
from backfill import ingest_dayone
from pathlib import Path
from datetime import datetime

df_dayone = ingest_dayone(Path('data/dayone/dayone.json'))
day = datetime(2023, 7, 6).date()
d_dayone = df_dayone[df_dayone['date'] == day]
print(f'DayOne entries for {day}: {len(d_dayone)}')

if len(d_dayone) > 0:
    print('Columns:', d_dayone.columns.tolist())
    for _, row in d_dayone.iterrows():
        ts = row['timestamp'].strftime('%H:%M')
        print(f'Time: {ts}')
        print(f'Place: {row.get("place_name", "None")}')
        print(f'Weather: {row.get("weather", "None")}')
        text = row.get('text', '')
        if text:
            print(f'Text: {text[:100]}...')
        print('---')
