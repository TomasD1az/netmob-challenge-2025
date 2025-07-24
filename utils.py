from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.mixture import GaussianMixture

# ——— configuration ————————————————————————————————————————————————————
PROBS_DIR       = Path('centroid_probs')       # where the probability CSVs will live
# ——————————————————————————————————————————————————————————————————————

# for csv_path in PROBS_DIR.glob('*.csv'):
#     df = pd.read_csv(csv_path)
#     prob_sum = df['probability'].sum()
#     df['probability'] = df['probability'] / prob_sum
#     df.to_csv(csv_path, index=False)
