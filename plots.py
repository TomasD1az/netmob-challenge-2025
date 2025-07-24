
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

def plot_user_points_and_centroid_probs(user_id, gps_dir='data/gps_filtrados', probs_dir='centroid_probs', centroids_file='grid_centroids.csv'):
    # Load user GPS points
    gps_path = Path(gps_dir) / f"{user_id}_filtrado.csv"
    df_gps = pd.read_csv(gps_path)

    # Load centroid probabilities
    if probs_dir == 'centroid_probs':
        probs_path = Path(probs_dir) / f"{user_id}_filtrado_centroid_probs.csv"
        df_probs = pd.read_csv(probs_path)
    elif probs_dir == 'predicted_probs':
        probs_path = Path(probs_dir) / f"{user_id}_predicted_probs.csv"
        df_probs = pd.read_csv(probs_path)
    elif probs_dir == 'final_predicted_probs':
        probs_path = Path(probs_dir) / f"{user_id}_predicted_probs.csv"
        df_probs = pd.read_csv(probs_path)
    
    # Plot user points
    plt.figure(figsize=(8, 8))
    plt.scatter(df_gps['LONGITUDE'], df_gps['LATITUDE'], s=10, c='blue', label='User Points', alpha=0.6)

    # Plot centroids with color intensity by probability
    norm_probs = df_probs['probability'] / df_probs['probability'].max() if df_probs['probability'].max() > 0 else df_probs['probability']
    plt.scatter(df_probs['LONGITUDE'], df_probs['LATITUDE'],
                s=80, c=norm_probs, cmap='Reds', label='Centroid Probabilities', alpha=0.8, edgecolor='k')

    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.title(f'User {user_id}: GPS Points & Centroid Probabilities')
    plt.legend()
    plt.colorbar(label='Relative Probability (centroids)')
    plt.tight_layout()
    plt.show()
    
def aggregate_and_plot(input_dir="final_predicted_probs", 
                       file_pattern="*_predicted_probs.csv",
                       figsize=(10, 8),
                       cmap="Reds",
                       point_size=80,
                       alpha=0.8,
                       edgecolor="k",
                       title="Aggregated & Globally Normalized Predictions"):
    import glob
    import pandas as pd
    import matplotlib.pyplot as plt

    # 1. Read all per-user CSVs
    csv_files = glob.glob(f"{input_dir}/{file_pattern}")
    if not csv_files:
        raise FileNotFoundError(f"No files matching {input_dir}/{file_pattern}")
    df_all = pd.concat((pd.read_csv(f) for f in csv_files), ignore_index=True)

    # 2. Aggregate duplicates by summing their probabilities
    df_agg = (
        df_all
        .groupby(['LATITUDE', 'LONGITUDE'], as_index=False)
        .agg({'probability': 'sum'})
    )

    # 3. Normalize globally to [0,1]
    p_min, p_max = df_agg['probability'].min(), df_agg['probability'].max()
    if p_max > p_min:
        df_agg['norm_prob'] = (df_agg['probability'] - p_min) / (p_max - p_min)
    else:
        df_agg['norm_prob'] = 0.0

    # 4. Single combined scatter‐plot
    plt.figure(figsize=figsize)
    plt.scatter(
        df_agg['LONGITUDE'],
        df_agg['LATITUDE'],
        s=point_size,
        c=df_agg['norm_prob'],
        cmap=cmap,
        edgecolor=edgecolor,
        alpha=alpha
    )
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.title(title)
    plt.colorbar(label='Normalized Probability')
    plt.tight_layout()
    plt.show()

    # Return the aggregated DataFrame if you want to inspect or save it
    return df_agg

aggregate_and_plot(input_dir="geriatric_predicted_probs")
aggregate_and_plot(input_dir="bmw_predicted_probs")
aggregate_and_plot(input_dir="unemployed_predicted_probs")

# Example usage:
# plot_user_points_and_centroid_probs('13_3560', probs_dir='final_predicted_probs')
# plot_user_points_and_centroid_probs('16_3918', probs_dir='final_predicted_probs')
# plot_user_points_and_centroid_probs('16_3999', probs_dir='final_predicted_probs')
# plot_user_points_and_centroid_probs('42_0159', probs_dir='final_predicted_probs')
# plot_user_points_and_centroid_probs('47_1065', probs_dir='final_predicted_probs')
# plot_user_points_and_centroid_probs('47_1149', probs_dir='final_predicted_probs')
# plot_user_points_and_centroid_probs('49_1569', probs_dir='final_predicted_probs')

# plot_user_points_and_centroid_probs('48_1372', probs_dir='predicted_probs')