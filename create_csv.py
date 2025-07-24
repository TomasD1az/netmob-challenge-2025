
from pathlib import Path
import pandas as pd
import numpy as np
from tqdm import tqdm

# ——— configuration ————————————————————————————————————————————————————
PROBS_DIR = Path('centroid_probs')       # directory with probability CSVs
NMF_FILE = Path('data/nmf_individuals_dataset.csv')       # file with NMF results
CLEAN_FILE = Path('data/individuals_dataset_cleaned.csv')  # file with clean dataset
# ——————————————————————————————————————————————————————————————————————

# Load NMF results (assume columns: ID, NMF_1, ..., NMF_10, CODGEO)
nmf_df = pd.read_csv(NMF_FILE)
nmf_df['ID'] = nmf_df['ID'].astype(str)

clean_df = pd.read_csv(CLEAN_FILE)
clean_df['ID'] = clean_df['ID'].astype(str)

all_rows = []

for csv_path in tqdm(list(PROBS_DIR.glob('*.csv')), desc="Processing users"):
    user_id = csv_path.stem.replace('_filtrado_centroid_probs', '')
    # Load centroid probabilities for this user
    probs_df = pd.read_csv(csv_path)
    # Find NMF row for this user
    nmf_row = nmf_df[nmf_df['ID'] == user_id]
    if nmf_row.empty:
        print(f"Warning: No NMF data for user {user_id}")
        continue
    nmf_row = nmf_row.iloc[0]
    
    clean_row = clean_df[clean_df['ID'] == user_id]
    if clean_row.empty:
        print(f"Warning: No clean data for user {user_id}")
        continue
    clean_row = clean_row.iloc[0]
    # For each centroid, create a row with all required columns
    for _, row in probs_df.iterrows():
        all_rows.append({
            'ID': user_id,
            'Latitude': row['LATITUDE'],
            'Longitude': row['LONGITUDE'],
            **{f'NMF_{i+1}': nmf_row[f'NMF_{i+1}'] for i in range(10)},
            'SEX': clean_row['SEX'],
            'AGE': clean_row['AGE'],
            'DIPLOMA': clean_row['DIPLOMA'],
            'PRO_CAT': clean_row['PRO_CAT'],
            'NBPERS_HOUSE': clean_row['NBPERS_HOUSE'],
            'NB_10': clean_row['NB_10'],
            'NB_11_17': clean_row['NB_11_17'],
            'NB_18_24': clean_row['NB_18_24'],
            'NB_25_64': clean_row['NB_25_64'],
            'NB_65': clean_row['NB_65'],
            'PMR': clean_row['PMR'],
            'DRIVING_LICENCE': clean_row['DRIVING_LICENCE'],
            'NB_CAR': clean_row['NB_CAR'],
            'TWO_WHEELER': clean_row['TWO_WHEELER'],
            'BIKE': clean_row['BIKE'],
            'ELECT_SCOOTER': clean_row['ELECT_SCOOTER'],
            'NAVIGO_SUB': clean_row['NAVIGO_SUB'],
            'IMAGINER_SUB': clean_row['IMAGINER_SUB'],
            'OTHER_SUB_PT': clean_row['OTHER_SUB_PT'],
            'BIKE_SUB': clean_row['BIKE_SUB'],  
            'NSM_SUB': clean_row['NSM_SUB'],
            'CODGEO': nmf_row['CODGEO'],
            'probability': row['probability']
        })

# Create final DataFrame
final_df = pd.DataFrame(all_rows)
print(final_df.head())
    
final_df.to_csv('dataset2.csv', index=False)
    
    