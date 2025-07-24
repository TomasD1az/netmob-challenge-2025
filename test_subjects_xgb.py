import joblib
import pandas as pd
from plots import plot_user_points_and_centroid_probs
import os
import matplotlib.pyplot as plt
import glob

def lr_schedule(round_idx):
    lr0 = 0.1
    decay_rate = 0.9
    decay_step = 50
    return lr0 * (decay_rate ** (round_idx // decay_step))

# Load trained model
model = joblib.load("xgb_final_model.pkl")

# Load new subjects data (replace 'new_subjects.csv' with your file)
new_data = pd.read_csv("dataset2/unemployed_df.csv")
old_data = pd.read_csv("data/individuals_dataset_cleaned.csv")

os.makedirs("unemployed_predicted_probs", exist_ok=True)

unique_ids = new_data['ID'].unique()

for id in unique_ids:
    # Find weight for this ID
    weight_row = old_data[old_data['ID'] == id]
    weight = weight_row.iloc[0]['WEIGHT_INDIV']

    # Select all rows for this user
    user_rows = new_data[new_data['ID'] == id]
    X_new = user_rows.drop(columns=['ID'])
    predictions = model.predict(X_new)
    weighted_preds = predictions * weight

    # Save all predictions for this user
    df_out = pd.DataFrame({
        'LATITUDE': user_rows['Latitude'].values,
        'LONGITUDE': user_rows['Longitude'].values,
        'probability': weighted_preds
    })
    out_path = f"unemployed_predicted_probs/{id}_predicted_probs.csv"
    df_out.to_csv(out_path, index=False)
    print(f"Saved weighted predictions for user {id} to {out_path}")
