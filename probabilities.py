"""
Compute probability of a user passing through each centroid using a Gaussian Mixture Model (GMM).

For every CSV in gps_dataset/ (one per user), the script:
  1. fits a 2-D GMM to the user's GPS points (lat, lon),
  2. evaluates the mixture PDF at every centroid from *grid_centroids.csv*,
  3. normalises the densities so they sum to 1 – interpreting them as discrete probabilities,
  4. saves the result to centroid_probs/<original_filename>_centroid_probs.csv
     with columns: latitude, longitude, probability
"""

from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.mixture import GaussianMixture
from sklearn.mixture._gaussian_mixture import _compute_precision_cholesky

# ——— configuration ————————————————————————————————————————————————————
DATA_DIR      = Path('data/gps_filtrados')          # raw or filtered gps points per user
CENTROID_FILE = Path('grid_centroids.csv')   # produced by centroids.py
OUT_DIR       = Path('centroid_probs')       # where the probability CSVs will live
MAX_COMPONENTS = 5                           # upper bound for GMM components to test
# ——————————————————————————————————————————————————————————————————————

OUT_DIR.mkdir(exist_ok=True)

# 0) Load centroids
centroids_df = pd.read_csv(CENTROID_FILE)
centroids_xy = centroids_df[['LONGITUDE', 'LATITUDE']].to_numpy()   # shape (n_centroids, 2)


def choose_best_gmm(X: np.ndarray, k_max: int = MAX_COMPONENTS) -> GaussianMixture:
    """Pick the number of components via BIC."""
    best_bic, best_gmm = np.inf, None
    max_components = min(k_max, len(X))
    if max_components < 1:
        raise ValueError("Not enough samples to fit a GMM.")
    for k in range(1, max_components+1):
        gmm = GaussianMixture(n_components=k, covariance_type='full', random_state=0)
        gmm.fit(X)
        bic = gmm.bic(X)
        if bic < best_bic:
            best_bic, best_gmm = bic, gmm
    return best_gmm


for csv_path in DATA_DIR.glob('*.csv'):
    df = pd.read_csv(csv_path)
    if df.empty:
        print(f"Skipping {csv_path.name}: empty file.")
        continue
    # rename/standardise column names if needed
    if {'LONGITUDE','LATITUDE'}.issubset(df.columns):
        lon, lat = df['LONGITUDE'].to_numpy(), df['LATITUDE'].to_numpy()
    else:
        raise ValueError(f'Columns latitude/longitude not found in {csv_path.name}')
    X = np.column_stack([lon, lat])

    # 1) fit GMM
    gmm = choose_best_gmm(X)
    inflation = 4.0    # e.g. double the standard deviation
    gmm.covariances_ *= inflation**2
    # recompute the precisions_cholesky so score_samples() still works:
    gmm.precisions_cholesky_ = _compute_precision_cholesky(
        gmm.covariances_, gmm.covariance_type
    )
    # 2) evaluate log-density at each centroid, convert to pdf
    logp = gmm.score_samples(centroids_xy)  # shape (n_centroids,)
    pdf  = np.exp(logp)

    # 3) normalise to probability mass
    probs = pdf / pdf.sum()

    # 4) save
    out = centroids_df.copy()
    out['probability'] = probs
    out_path = OUT_DIR / f"{csv_path.stem}_centroid_probs.csv"
    out.to_csv(out_path, index=False)
    print(f"{csv_path.name:30s} → {out_path.name:30s}  (components: {gmm.n_components})")