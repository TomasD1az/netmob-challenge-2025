import argparse
import pandas as pd
import numpy as np
import time
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import (
    GroupShuffleSplit,
    GroupKFold,
    RandomizedSearchCV
)
from sklearn.metrics import mean_squared_error
import xgboost as xgb
import joblib
from xgboost.callback import LearningRateScheduler

def load_and_prepare(path, target_col, user_col):
    print(f"[{time.strftime('%H:%M:%S')}] Cargando datos desde {path}…")
    df = pd.read_csv(path)
    for col in (target_col, user_col):
        if col not in df.columns:
            raise ValueError(f"Columna '{col}' no encontrada en el CSV.")
    groups = df[user_col].values
    y = df[target_col].values
    X = df.drop(columns=[target_col, user_col])
    for c in X.select_dtypes(include=["object", "category"]):
        X[c] = LabelEncoder().fit_transform(X[c].astype(str))
    print(f"[{time.strftime('%H:%M:%S')}] Datos preparados: {X.shape[0]} filas × {X.shape[1]} features")
    return X, y, groups

def get_param_dist():
    return {
        "n_estimators":     [100, 300, 500, 800],
        "max_depth":        [3, 5, 7, 9],
        "learning_rate":    [0.01, 0.05, 0.1, 0.2],
        "subsample":        [0.6, 0.8, 1.0],
        "colsample_bytree": [0.6, 0.8, 1.0],
        "gamma":            [0, 1, 5],
        "reg_alpha":        [0, 0.1, 1],
        "reg_lambda":       [1, 5, 10],
    }

def lr_schedule(round_idx):
    """Decay exponencial cada 50 rounds."""
    lr0 = 0.1
    decay_rate = 0.9
    decay_step = 50
    return lr0 * (decay_rate ** (round_idx // decay_step))

def main():
    p = argparse.ArgumentParser(
        description="XGBoost full pipeline con prints y tiempos"
    )
    p.add_argument("--data",      "-d", required=True, help="Ruta al CSV")
    p.add_argument("--target",    "-t", required=True, help="Columna target")
    p.add_argument("--user-col",  "-u", required=True, help="Columna user_id")
    p.add_argument("--test-size", type=float, default=0.2, help="Frac. usuarios para test")
    p.add_argument("--n-splits",  type=int,   default=5,   help="Folds para GroupKFold")
    p.add_argument("--n-iter",    type=int,   default=50,  help="Iter. RandomSearch")
    p.add_argument("--seed",      type=int,   default=42,  help="Random seed")
    args = p.parse_args()

    # 1) Carga y preprocesamiento
    X, y, groups = load_and_prepare(args.data, args.target, args.user_col)

    # 2) Hold-out inicial por usuario
    print(f"[{time.strftime('%H:%M:%S')}] Creando hold-out test ({args.test_size*100:.1f}% usuarios)…")
    gss = GroupShuffleSplit(
        n_splits=1,
        test_size=args.test_size,
        random_state=args.seed
    )
    train_idx, test_idx = next(gss.split(X, y, groups))
    X_train, y_train = X.iloc[train_idx], y[train_idx]
    X_test,  y_test  = X.iloc[test_idx],  y[test_idx]
    groups_train   = groups[train_idx]
    print(f"[{time.strftime('%H:%M:%S')}] Train: {X_train.shape[0]} filas, Test: {X_test.shape[0]} filas")

    # 3) Configuración de búsqueda
    base_model = xgb.XGBRegressor(
        objective="reg:squarederror",
        tree_method="hist",
        verbosity=1,               # logs por round
        random_state=args.seed
    )
    param_dist = get_param_dist()
    cv = GroupKFold(n_splits=args.n_splits)

    search = RandomizedSearchCV(
    estimator=base_model,
    param_distributions=param_dist,
    n_iter=args.n_iter,
    scoring="r2",            # ⬅️ CAMBIADO: ahora optimiza R²
    cv=cv,
    verbose=2,
    n_jobs=-1,
    random_state=args.seed,
    refit=True
)


    # 4) Búsqueda de hiperparámetros
    t0 = time.time()
    print(f"[{time.strftime('%H:%M:%S')}] Iniciando hyperparameter search (n_iter={args.n_iter})…")
    search.fit(X_train, y_train, groups=groups_train)
    t_search = time.time() - t0
    print(f"[{time.strftime('%H:%M:%S')}] Hyperparameter search completado en {t_search/60:.2f} min")

    best_params = search.best_params_
    cv_r2 = search.best_score_     # ⬅️ R² ya está en positivo
    print(f"→ Mejores parámetros CV: {best_params}")
    with open("xgb_best_params.txt", "w") as f:
        for k, v in best_params.items():
            f.write(f"{k}: {v}\n")
    print(f"→ Mejores parámetros guardados en 'xgb_best_params.txt'")
    print(f"→ R² (CV agrupado): {cv_r2:.4f}")


    # 5) Entrenamiento final con scheduler
    print(f"[{time.strftime('%H:%M:%S')}] Retraining final model con scheduler…")
    final_model = xgb.XGBRegressor(
        **best_params,
        objective="reg:squarederror",
        tree_method="hist",
        verbosity=1,
        random_state=args.seed
    )
    t1 = time.time()
    final_model.fit(
        X_train, y_train,
        callbacks=[LearningRateScheduler(lr_schedule)],
        verbose=False
    )
    t_final = time.time() - t1
    print(f"[{time.strftime('%H:%M:%S')}] Entrenamiento final completado en {t_final:.2f} seg")

    # 6) Evaluación en hold-out
    preds = final_model.predict(X_test)
    rmse_test = mean_squared_error(y_test, preds, squared=False)
    print(f"\nRMSE en test final: {rmse_test:.4f}")

    # 7) Guardar modelo
    joblib.dump(final_model, "xgb_final_model.pkl")
    print(f"[{time.strftime('%H:%M:%S')}] Modelo guardado en 'xgb_final_model.pkl'")

if __name__ == "__main__":
    main()
