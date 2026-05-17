import sys
import os
from pathlib import Path
import pandas as pd
import soundfile as sf
import io

out_dir = Path('tests/input-audio-files/commonphone_test')
out_dir.mkdir(parents=True, exist_ok=True)

urls = [
    'https://huggingface.co/datasets/pklumpp/CommonPhoneDataset/resolve/refs%2Fconvert%2Fparquet/default/test/0000.parquet',
    'https://huggingface.co/datasets/pklumpp/CommonPhoneDataset/resolve/refs%2Fconvert%2Fparquet/default/test/0005.parquet',
    'https://huggingface.co/datasets/pklumpp/CommonPhoneDataset/resolve/refs%2Fconvert%2Fparquet/default/test/0015.parquet',
    'https://huggingface.co/datasets/pklumpp/CommonPhoneDataset/resolve/refs%2Fconvert%2Fparquet/default/test/0018.parquet',
]

target_per_age = 3
downloaded_counts = {}
metadata = []
total_to_download = 20

def _safe_str(v): return str(v).lower().replace(' ', '_')

print('Starting selective download...')
for url in urls:
    if len(metadata) >= total_to_download: break
    print(f'Reading {url.split("/")[-1]}...')
    try:
        df = pd.read_parquet(url)
    except Exception as e:
        print(f"Failed to read {url}: {e}")
        continue
    
    # Shuffle
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    for i, row in df.iterrows():
        age = _safe_str(row.get('age', 'unknown'))
        if downloaded_counts.get(age, 0) >= target_per_age:
            continue
            
        downloaded_counts[age] = downloaded_counts.get(age, 0) + 1
        
        audio_bytes = row['audio']['bytes']
        array, sr = sf.read(io.BytesIO(audio_bytes))
        
        gender = _safe_str(row.get('gender', 'unknown'))
        cid = _safe_str(row.get('client_id', f'{i}'))[:12]
        fname = f'{cid}_{gender}_{age}.wav'
        
        out_path = out_dir / fname
        sf.write(str(out_path), array, sr, subtype='PCM_16')
        
        metadata.append({
            'filename': fname,
            'client_id': row.get('client_id'),
            'gender': row.get('gender'),
            'age': row.get('age'),
            'locale': row.get('locale'),
            'sentence': row.get('sentence'),
        })
        
        print(f'Saved [{len(metadata)}/{total_to_download}] {fname}')
        if len(metadata) >= total_to_download:
            break

with open(out_dir / 'metadata.tsv', 'w', encoding='utf-8') as f:
    f.write('filename\tgender\tage\n')
    for m in metadata:
        f.write(str(m['filename']) + '\t' + str(m['gender']) + '\t' + str(m['age']) + '\n')

print('Done. Age distribution:', downloaded_counts)
