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
from sklearn.metrics import mean_squared_error, root_mean_squared_error, r2_score
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
    # X = df.drop(columns=[target_col, user_col, 'NMF_1', 'NMF_2', 'NMF_3', 'NMF_4', 'NMF_5', 'NMF_6', 'NMF_7', 'NMF_8', 'NMF_9', 'NMF_10'])
    for c in X.select_dtypes(include=["object", "category"]):
        X[c] = LabelEncoder().fit_transform(X[c].astype(str))
    print(f"[{time.strftime('%H:%M:%S')}] Datos preparados: {X.shape[0]} filas × {X.shape[1]} features")
    return X, y, groups

def lr_schedule(round_idx):
    """Decay exponencial cada 50 rounds."""
    lr0 = 0.1
    decay_rate = 0.9
    decay_step = 50
    return lr0 * (decay_rate ** (round_idx // decay_step))

# def weighted_mse_obj(y_true, y_pred):
#     # elige cuánto más quieres penalizar los no-ceros
#     weight_factor = 100

#     # peso 1.0 para y_true == 0, weight_factor para y_true != 0
#     w = np.where(y_true == 0, 0.001, weight_factor)

#     # gradiente y hessiano de (y_pred - y_true)**2 * w
#     grad = 2 * (y_pred - y_true) * w
#     hess = 2 * w

#     return grad, hess

def weighted_sqerror_obj(y_true, y_pred):
    """
    Weighted squared‐error loss: L = sum_i w_i * (y_pred_i - y_true_i)^2
    so that grad_i = 2 w_i (y_pred_i - y_true_i),
            hess_i = 2 w_i.
    """
    # define your class weights:
    weight_zero    = 1   # weight for y_true == 0
    weight_nonzero = 1   # weight for y_true != 0

    # build the per‐sample weights
    w = np.where(y_true == 0, weight_zero, weight_nonzero)

    # gradient and hessian of w*(pred - true)^2
    grad = 2 * w * (y_pred - y_true)
    hess = 2 * w
    return grad, hess

def save_all_user_predictions(X_test, preds, groups, output_dir="predicted_probs"):
    import os
    os.makedirs(output_dir, exist_ok=True)
    unique_users = np.unique(groups)
    for user_id in unique_users:
        user_mask = (groups == user_id)
        test_indices = np.where(user_mask)[0]
        df_out = pd.DataFrame({
            "LATITUDE": X_test["Latitude"].iloc[test_indices].values,
            "LONGITUDE": X_test["Longitude"].iloc[test_indices].values,
            "probability": preds[test_indices]
        })
        out_path = f"{output_dir}/{user_id}_predicted_probs.csv"
        df_out.to_csv(out_path, index=False)
        print(f"Saved predictions for user {user_id} to {out_path}")

def main():
    p = argparse.ArgumentParser(
        description="XGBoost full pipeline con prints y tiempos"
    )
    p.add_argument("--data",      "-d", required=True, help="Ruta al CSV")
    p.add_argument("--target",    "-t", required=True, help="Columna target")
    p.add_argument("--user-col",  "-u", required=True, help="Columna user_id")
    p.add_argument("--test-size", type=float, default=0.2, help="Frac. usuarios para test")
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
    groups_test      = groups[test_idx]
    print(f"[{time.strftime('%H:%M:%S')}] Train: {X_train.shape[0]} filas, Test: {X_test.shape[0]} filas")

    # 3) Configuración de búsqueda

    best_params = {
        'subsample': 1,
        'reg_lambda': 10,
        'reg_alpha': 1,
        'n_estimators': 500,
        'max_depth': 7,
        'learning_rate': 0.05,
        'gamma': 0,
        'colsample_bytree': 1.0
    }

    # 5) Entrenamiento final con scheduler
    print(f"[{time.strftime('%H:%M:%S')}] Retraining final model con scheduler…")

    final_model = xgb.XGBRegressor(
        **best_params,
        # objective=weighted_sqerror_obj, 
        tree_method="hist",
        verbosity=1,
        random_state=args.seed,
        eval_metric="rmse",
        callbacks=[LearningRateScheduler(lr_schedule)]
    )

    t1 = time.time()

    # y luego en tu fit:
    final_model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        verbose=True
    )
    
    t_final = time.time() - t1
    print(f"[{time.strftime('%H:%M:%S')}] Entrenamiento final completado en {t_final:.2f} seg")

    # 6) Evaluación en test
    preds = final_model.predict(X_test)
    mse = mean_squared_error(y_test, preds)
    rmse_test = np.sqrt(mse)
    print(f"RMSE en test final: {rmse_test:.4f}")
    
    # 2) R²
    r2 = r2_score(y_test, preds)
    print(f"R² en test final: {r2:.4f}")
    
    # 7) Guardar modelo
    joblib.dump(final_model, "xgb_final_model.pkl")
    print(f"[{time.strftime('%H:%M:%S')}] Modelo guardado en 'xgb_final_model.pkl'")

    # 8) Guardar predicciones por usuario
    save_all_user_predictions(X_test, preds, groups_test)

if __name__ == "__main__":
    main()
    # After main, load test predictions and groups
    # You may need to adjust this if you want to run after training
