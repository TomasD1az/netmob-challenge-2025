import glob
import pandas as pd
import numpy as np

# 1) Load all CSVs and collect all lon/lat
paths = glob.glob("data/gps_filtrados/*.csv")
lons, lats = [], []
for p in paths:
    df = pd.read_csv(p)
    # adjust column names as needed
    lons.extend(df['LONGITUDE'])
    lats.extend(df['LATITUDE'])

# 2) Compute raw bbox
min_lon, max_lon = min(lons), max(lons)
min_lat, max_lat = min(lats), max(lats)

# 3) Pad bbox into a square
dx = max_lon - min_lon
dy = max_lat - min_lat
if dx > dy:
    pad = (dx - dy) / 2
    min_lat -= pad
    max_lat += pad
    side = dx
else:
    pad = (dy - dx) / 2
    min_lon -= pad
    max_lon += pad
    side = dy

# 4) Compute size of each grid cell
n = 50
cell_size = side / n

# 5) Generate centroids
centroids = []
for i in range(n):
    for j in range(n):
        cx = min_lon + (i + 0.5) * cell_size
        cy = min_lat + (j + 0.5) * cell_size
        centroids.append((cy, cx))   # (lat, lon)

# 6) Save centroids to DataFrame / CSV
cent_df = pd.DataFrame(centroids, columns=['LATITUDE', 'LONGITUDE'])
cent_df.to_csv("grid_centroids.csv", index=False)

print(f"Generated {len(cent_df)} centroids and wrote to grid_centroids.csv")