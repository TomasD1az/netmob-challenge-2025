import pandas as pd
import time
from sklearn.model_selection import (
    GroupShuffleSplit,
    GroupKFold,
    RandomizedSearchCV
)

def load_and_prepare(path, target_col, user_col):
    print(f"[{time.strftime('%H:%M:%S')}] Cargando datos desde {path}…")
    df = pd.read_csv(path)
    for col in (target_col, user_col):
        if col not in df.columns:
            raise ValueError(f"Columna '{col}' no encontrada en el CSV.")
    groups = df[user_col].values
    y = df[target_col].values
    X = df.drop(columns=[target_col])
    print(f"[{time.strftime('%H:%M:%S')}] Datos preparados: {X.shape[0]} filas × {X.shape[1]} features")
    return X, y, groups

X, y, groups = load_and_prepare("dataset2\dataset.csv", "probability", "ID")

gss = GroupShuffleSplit(
        n_splits=1,
        test_size=0.15,
        random_state=42
    )
train_idx, test_idx = next(gss.split(X, y, groups))
X_train, y_train = X.iloc[train_idx], y[train_idx]
X_test,  y_test  = X.iloc[test_idx],  y[test_idx]

def look_test(eq_parameters: dict, gte_parameters: dict, lte_parameters: dict, test_Set: pd.DataFrame) -> pd.DataFrame:
    # Start with a mask that selects all rows
    mask = pd.Series(True, index=test_Set.index)

    # Apply equality filters
    for col, val in eq_parameters.items():
        if col in test_Set.columns:
            mask &= (test_Set[col] == val)

    # Apply greater-than-or-equal filters
    for col, val in gte_parameters.items():
        if col in test_Set.columns:
            mask &= (test_Set[col] >= val)

    # Apply less-than-or-equal filters
    for col, val in lte_parameters.items():
        if col in test_Set.columns:
            mask &= (test_Set[col] <= val)

    return test_Set[mask]

BMW_eq = {"SEX": 1.0, "PRO_CAT": 2.0, "DRIVING_LICENSE": 1.0, "PMR": 0.0, "NAVIGO_SUB": 0.0, "IMAGINER_SUB": 0.0, "OTHER_SUB_PT": 0.0, "BIKE_SUB": 0.0, "NSM_SUB":0.0}
BMW_gte = {"AGE": 50.0, "DIPLOMA": 4.0}

BMW_df = look_test(
    eq_parameters=BMW_eq,
    gte_parameters=BMW_gte,
    lte_parameters={},
    test_Set=X_test
)
BMW_df.drop(columns=["probability", "Latitude", "Longitude"], inplace=True, errors='ignore')
BMW_df = BMW_df.drop_duplicates()
BMW_df.to_csv("dataset2/BMW_df.csv", index=False)

geriatric_eq = {}
geriatric_gte = {"AGE": 30, "NB_65": 1.0, "NBPERS_HOUSE": 2.0}
geriatric_lte = {"PRO_CAT": 5}

geriatric_df = look_test(
    eq_parameters=geriatric_eq,
    gte_parameters=geriatric_gte,
    lte_parameters=geriatric_lte,
    test_Set=X_test
)
geriatric_df.drop(columns=["probability", "Latitude", "Longitude"], inplace=True, errors='ignore')
geriatric_df = geriatric_df.drop_duplicates()
geriatric_df.to_csv("dataset2/geriatric_df.csv", index=False)

unemployed_eq = {"DRIVING_LICENSE": 0.0}
unemployed_gte = {"AGE": 18.0, "PRO_CAT": 7.0}
unemployed_lte = {"AGE": 25.0, "DIPLOMA": 2.0}

unemployed_df = look_test(
    eq_parameters=unemployed_eq,
    gte_parameters=unemployed_gte,
    lte_parameters=unemployed_lte,
    test_Set=X_test
)
unemployed_df.drop(columns=["probability", "Latitude", "Longitude"], inplace=True, errors='ignore')
unemployed_df = unemployed_df.drop_duplicates()
unemployed_df.to_csv("dataset2/unemployed_df.csv", index=False)