import logging
import json
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import dump, load

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    roc_auc_score,
    average_precision_score,
)
from sklearn.feature_selection import mutual_info_classif

from xgboost import XGBClassifier


class Config:
    RANDOM_STATE = 42
    TOP_K_FEATURES = 200
    VALID_SIZE_SUBJECTS = 0.2
    MODEL_DIR = Path("artifacts")
    CSV_PATH = Path("ml_ready_balanced.csv")

    XGB_PARAM_GRID = [
        {
            "n_estimators": 300,
            "max_depth": 4,
            "learning_rate": 0.05,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
        },
        {
            "n_estimators": 500,
            "max_depth": 4,
            "learning_rate": 0.05,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
        },
        {
            "n_estimators": 500,
            "max_depth": 6,
            "learning_rate": 0.05,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
        },
        {
            "n_estimators": 700,
            "max_depth": 6,
            "learning_rate": 0.03,
            "subsample": 0.9,
            "colsample_bytree": 0.9,
        },
    ]


logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] %(message)s",
)


def load_ecg_data(csv_path: Path) -> pd.DataFrame:
    logging.info(f"Loading data from {csv_path}")
    df = pd.read_csv(csv_path)
    ecg_df = df[df["Modality"] == "ECG"].copy()
    logging.info(f"Total ECG rows: {len(ecg_df)}")
    logging.info("Label distribution (ECG):\n%s", ecg_df["Label"].value_counts())
    return ecg_df


def subject_wise_split(ecg_df: pd.DataFrame, valid_size_subjects: float, random_state: int):
    train_subjects = ecg_df.loc[ecg_df["split"] == "train", "Subject_ID"].unique()
    test_subjects = ecg_df.loc[ecg_df["split"] == "test", "Subject_ID"].unique()

    train_subj, val_subj = train_test_split(
        train_subjects,
        test_size=valid_size_subjects,
        random_state=random_state,
        shuffle=True,
    )

    train_df = ecg_df[ecg_df["Subject_ID"].isin(train_subj)].copy()
    val_df   = ecg_df[ecg_df["Subject_ID"].isin(val_subj)].copy()
    test_df  = ecg_df[ecg_df["Subject_ID"].isin(test_subjects)].copy()

    logging.info("Train subjects: %d", len(np.unique(train_df["Subject_ID"])))
    logging.info("Val subjects:   %d", len(np.unique(val_df["Subject_ID"])))
    logging.info("Test subjects:  %d", len(np.unique(test_df["Subject_ID"])))

    return train_df, val_df, test_df


METADATA_COLS = [
    "Subject_ID",
    "Modality",
    "split",
    "Onset_Seconds",
    "Duration_Seconds",
    "Epoch_Index",
]

def drop_metadata(train_df, val_df, test_df):
    for col in METADATA_COLS:
        if col in train_df.columns:
            train_df = train_df.drop(columns=[col])
            val_df   = val_df.drop(columns=[col])
            test_df  = test_df.drop(columns=[col])
    return train_df, val_df, test_df


def encode_and_scale(train_df, val_df, test_df, random_state: int):
    y_train = train_df["Label"].values
    y_val   = val_df["Label"].values
    y_test  = test_df["Label"].values

    X_train = train_df.drop(columns=["Label"])
    X_val   = val_df.drop(columns=["Label"])
    X_test  = test_df.drop(columns=["Label"])

    categorical_cols = []
    if "Sex" in X_train.columns:
        categorical_cols.append("Sex")

    X_train_enc = pd.get_dummies(X_train, columns=categorical_cols, drop_first=True)
    X_val_enc   = pd.get_dummies(X_val,   columns=categorical_cols, drop_first=True)
    X_test_enc  = pd.get_dummies(X_test,  columns=categorical_cols, drop_first=True)

    X_val_enc  = X_val_enc.reindex(columns=X_train_enc.columns, fill_value=0)
    X_test_enc = X_test_enc.reindex(columns=X_train_enc.columns, fill_value=0)

    feature_names = list(X_train_enc.columns)
    logging.info("Feature dimension after encoding: %d", len(feature_names))

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_enc)
    X_val_scaled   = scaler.transform(X_val_enc)
    X_test_scaled  = scaler.transform(X_test_enc)

    return (
        X_train_scaled,
        X_val_scaled,
        X_test_scaled,
        y_train,
        y_val,
        y_test,
        feature_names,
        scaler,
    )


def feature_selection_mi(X_train, y_train, X_val, X_test, top_k: int):
    logging.info("Running mutual information feature selection...")
    mi = mutual_info_classif(X_train, y_train, random_state=Config.RANDOM_STATE)
    idx_sorted = np.argsort(mi)[::-1]
    k = min(top_k, X_train.shape[1])
    top_idx = idx_sorted[:k]

    X_train_sel = X_train[:, top_idx]
    X_val_sel   = X_val[:, top_idx]
    X_test_sel  = X_test[:, top_idx]

    logging.info("Selected top %d features.", X_train_sel.shape[1])
    return X_train_sel, X_val_sel, X_test_sel, top_idx, mi[top_idx]


def evaluate_threshold(y_true, y_scores, threshold):
    y_pred = (y_scores >= threshold).astype(int)
    acc = accuracy_score(y_true, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", zero_division=0
    )
    return acc, prec, rec, f1


def find_best_threshold(y_true, y_scores, metric="f1"):
    best_thr = 0.5
    best_score = -1.0
    best_tuple = None

    thresholds = np.linspace(0.1, 0.9, 17)
    for thr in thresholds:
        acc, prec, rec, f1 = evaluate_threshold(y_true, y_scores, thr)
        score = {"f1": f1, "recall": rec, "precision": prec, "accuracy": acc}[metric]
        if score > best_score:
            best_score = score
            best_thr = thr
            best_tuple = (acc, prec, rec, f1)

    logging.info(
        "Best threshold on val (metric=%s): %.3f -> Acc=%.3f, Prec=%.3f, Rec=%.3f, F1=%.3f",
        metric,
        best_thr,
        best_tuple[0],
        best_tuple[1],
        best_tuple[2],
        best_tuple[3],
    )

    return best_thr, best_tuple


def train_and_select_xgb(
    X_train,
    y_train,
    X_val,
    y_val,
    param_grid,
    random_state: int,
):
    best_model = None
    best_params = None
    best_thr = 0.5
    best_val_f1 = -1.0
    best_val_metrics = None

    for i, params in enumerate(param_grid):
        logging.info("Training XGBoost config %d/%d: %s", i + 1, len(param_grid), params)

        model = XGBClassifier(
            objective="binary:logistic",
            eval_metric="logloss",
            tree_method="hist",
            n_jobs=-1,
            random_state=Config.RANDOM_STATE,
            **params,
        )
        model.fit(X_train, y_train)

        y_val_scores = model.predict_proba(X_val)[:, 1]

        thr, (acc, prec, rec, f1) = find_best_threshold(y_val, y_val_scores, metric="f1")

        logging.info(
            "Config %d -> Val F1=%.3f (thr=%.3f, Acc=%.3f, Prec=%.3f, Rec=%.3f)",
            i + 1,
            f1,
            thr,
            acc,
            prec,
            rec,
        )

        if f1 > best_val_f1:
            best_val_f1 = f1
            best_model = model
            best_params = params
            best_thr = thr
            best_val_metrics = (acc, prec, rec, f1)

    logging.info("Best XGBoost params: %s", best_params)
    logging.info(
        "Best val F1: %.3f at threshold %.3f (Acc=%.3f, Prec=%.3f, Rec=%.3f)",
        best_val_f1,
        best_thr,
        best_val_metrics[0],
        best_val_metrics[1],
        best_val_metrics[2],
    )

    return best_model, best_params, best_thr


def evaluate_on_test(model, threshold, X_test, y_test):
    y_scores = model.predict_proba(X_test)[:, 1]
    acc, prec, rec, f1 = evaluate_threshold(y_test, y_scores, threshold)

    try:
        roc = roc_auc_score(y_test, y_scores)
        pr_auc = average_precision_score(y_test, y_scores)
    except Exception:
        roc = np.nan
        pr_auc = np.nan

    logging.info(
        "TEST -> Thr=%.3f | Acc=%.3f, Prec=%.3f, Rec=%.3f, F1=%.3f, ROC-AUC=%.3f, PR-AUC=%.3f",
        threshold,
        acc,
        prec,
        rec,
        f1,
        roc,
        pr_auc,
    )

    return {
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "roc_auc": roc,
        "pr_auc": pr_auc,
    }


def save_artifacts(
    model,
    scaler,
    feature_names,
    selected_indices,
    threshold,
    metrics,
    config: Config,
):
    MODEL_DIR = config.MODEL_DIR
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    model_path = MODEL_DIR / "xgb_ecg_seizure_model.joblib"
    scaler_path = MODEL_DIR / "scaler.joblib"
    info_path = MODEL_DIR / "model_info.json"

    dump(model, model_path)
    dump(scaler, scaler_path)

    info = {
        "threshold": float(threshold),
        "selected_feature_indices": selected_indices.tolist(),
        "feature_names": feature_names,
        "test_metrics": metrics,
        "config": {
            "TOP_K_FEATURES": config.TOP_K_FEATURES,
            "VALID_SIZE_SUBJECTS": config.VALID_SIZE_SUBJECTS,
            "RANDOM_STATE": config.RANDOM_STATE,
        },
    }

    with open(info_path, "w", encoding="utf-8") as f:
        json.dump(info, f, indent=2)

    logging.info("Saved model to %s", model_path)
    logging.info("Saved scaler to %s", scaler_path)
    logging.info("Saved model info to %s", info_path)


def main():
    cfg = Config()
    np.random.seed(cfg.RANDOM_STATE)

    ecg_df = load_ecg_data(cfg.CSV_PATH)

    train_df, val_df, test_df = subject_wise_split(
        ecg_df, valid_size_subjects=cfg.VALID_SIZE_SUBJECTS, random_state=cfg.RANDOM_STATE
    )

    train_df, val_df, test_df = drop_metadata(train_df, val_df, test_df)

    (
        X_train_scaled,
        X_val_scaled,
        X_test_scaled,
        y_train,
        y_val,
        y_test,
        feature_names,
        scaler,
    ) = encode_and_scale(train_df, val_df, test_df, random_state=cfg.RANDOM_STATE)

    (
        X_train_sel,
        X_val_sel,
        X_test_sel,
        selected_idx,
        selected_mi,
    ) = feature_selection_mi(
        X_train_scaled,
        y_train,
        X_val_scaled,
        X_test_scaled,
        top_k=cfg.TOP_K_FEATURES,
    )

    best_model, best_params, best_thr = train_and_select_xgb(
        X_train_sel,
        y_train,
        X_val_sel,
        y_val,
        cfg.XGB_PARAM_GRID,
        random_state=cfg.RANDOM_STATE,
    )

    test_metrics = evaluate_on_test(
        best_model,
        best_thr,
        X_test_sel,
        y_test,
    )

    selected_feature_names = [feature_names[i] for i in selected_idx]
    save_artifacts(
        model=best_model,
        scaler=scaler,
        feature_names=selected_feature_names,
        selected_indices=selected_idx,
        threshold=best_thr,
        metrics=test_metrics,
        config=cfg,
    )

    logging.info("Training pipeline completed.")

if __name__ == "__main__":
    main()