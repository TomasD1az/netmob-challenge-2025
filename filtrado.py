import geopandas as gpd
import pandas as pd
from shapely.geometry import Point
from pathlib import Path


# 1) Rutas (ajustadas para la carpeta 'data')
shapefile_path = Path("data/arrondissements.shp")
input_folder    = Path("data/gps_dataset")
output_folder   = Path("data/gps_filtrados")
output_folder.mkdir(exist_ok=True)

# 2) Cargo polígonos y aseguro CRS WGS84
# ... (todo igual hasta cargar 'arr')
arr = gpd.read_file(shapefile_path).to_crs(epsg=4326)
print("Campos en shapefile:", arr.columns.tolist())

for csv_path in input_folder.glob("*.csv"):
    df = pd.read_csv(csv_path, parse_dates=["UTC DATETIME", "LOCAL DATETIME"])
    pts = gpd.GeoDataFrame(
        df,
        geometry=gpd.points_from_xy(df["LONGITUDE"], df["LATITUDE"]),
        crs="EPSG:4326"
    )

    # Usamos 'l_ar' para el nombre
    pts_in = gpd.sjoin(
        pts,
        arr[["geometry", "l_ar"]],
        how="inner",
        predicate="within"
    )

    pts_in = pts_in.drop(columns="geometry")
    if len(pts_in) > 5:
        out_path = output_folder / f"{csv_path.stem}_filtrado.csv"
        pts_in.to_csv(out_path, index=False)
        print(f"{csv_path.name} → {len(pts_in)} puntos dentro → {out_path.name}")
    else:
        print(f"{csv_path.name} → {len(pts_in)} puntos dentro (no se guarda archivo)")