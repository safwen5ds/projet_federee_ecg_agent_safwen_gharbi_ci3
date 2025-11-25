import logging
import json
import sys
import warnings
from pathlib import Path
from datetime import datetime
from typing import Tuple, Dict, Any, List

import numpy as np
import pandas as pd
from joblib import dump
from sklearn.model_selection import StratifiedGroupKFold, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    roc_auc_score,
    average_precision_score,
    f1_score,
    make_scorer
)
from sklearn.feature_selection import mutual_info_classif
from xgboost import XGBClassifier

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

# =========================
# CONFIG
# =========================

class Config:
    RANDOM_STATE = 42
    TOP_K_FEATURES = 200          # number of features to keep after MI selection
    N_FOLDS = 5                   # Number of CV folds
    N_ITER_SEARCH = 20            # Number of iterations for random search
    MODEL_DIR = Path("artifacts") # where to save model + scaler + config
    CSV_PATH = Path("ml_ready_balanced.csv")
    LOG_FILE = "training.log"

    # Hyperparameter search space for XGBoost
    XGB_PARAM_DIST = {
        "n_estimators": [100, 200, 300, 500, 700],
        "max_depth": [3, 4, 5, 6, 8, 10],
        "learning_rate": [0.01, 0.03, 0.05, 0.1, 0.2],
        "subsample": [0.6, 0.7, 0.8, 0.9, 1.0],
        "colsample_bytree": [0.6, 0.7, 0.8, 0.9, 1.0],
        "gamma": [0, 0.1, 0.2, 0.5],
        "min_child_weight": [1, 3, 5, 7],
    }

# =========================
# LOGGING
# =========================

def setup_logging():
    """Configure logging to both file and console."""
    # Create a custom logger
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    # Create handlers
    c_handler = logging.StreamHandler(sys.stdout)
    f_handler = logging.FileHandler(Config.LOG_FILE, mode='w')

    # Create formatters and add it to handlers
    log_format = logging.Formatter("[%(asctime)s] [%(levelname)s] %(message)s")
    c_handler.setFormatter(log_format)
    f_handler.setFormatter(log_format)

    # Add handlers to the logger
    # Remove existing handlers to avoid duplicates if re-run
    if logger.hasHandlers():
        logger.handlers.clear()
    
    logger.addHandler(c_handler)
    logger.addHandler(f_handler)
    return logger

logger = setup_logging()

# =========================
# DATA LOADING & PREPROCESSING
# =========================

def load_ecg_data(csv_path: Path) -> pd.DataFrame:
    """Load and filter ECG data."""
    try:
        logger.info(f"Loading data from {csv_path}")
        if not csv_path.exists():
            raise FileNotFoundError(f"File not found: {csv_path}")
            
        df = pd.read_csv(csv_path)
        ecg_df = df[df["Modality"] == "ECG"].copy()
        
        if ecg_df.empty:
            raise ValueError("No ECG data found in the CSV.")
            
        logger.info(f"Total ECG rows: {len(ecg_df)}")
        logger.info(f"Label distribution (ECG):\n{ecg_df['Label'].value_counts()}")
        return ecg_df
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        raise

def preprocess_data(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series, pd.Series]:
    """
    Drop metadata, handle categorical features, and prepare X, y, groups.
    """
    logger.info("Preprocessing data...")
    
    # Metadata columns to drop
    metadata_cols = [
        "Modality", "split", "Onset_Seconds", "Duration_Seconds", "Epoch_Index"
    ]
    
    # Extract groups (Subject_ID) before dropping
    groups = df["Subject_ID"]
    
    # Drop metadata
    X = df.drop(columns=["Label", "Subject_ID"] + [c for c in metadata_cols if c in df.columns])
    y = df["Label"]
    
    # Categorical encoding (e.g., Sex)
    if "Sex" in X.columns:
        X = pd.get_dummies(X, columns=["Sex"], drop_first=True)
        
    logger.info(f"Feature dimension: {X.shape[1]}")
    return X, y, groups

# =========================
# FEATURE SELECTION & SCALING
# =========================

def select_features_and_scale(X_train, y_train, X_val=None, top_k=200):
    """
    Select top k features using Mutual Information and scale data.
    Returns scaler, selected_indices, and transformed data.
    """
    # 1. Feature Selection
    logger.info(f"Running MI feature selection (top {top_k})...")
    mi = mutual_info_classif(X_train, y_train, random_state=Config.RANDOM_STATE)
    selected_indices = np.argsort(mi)[::-1][:top_k]
    
    X_train_sel = X_train.iloc[:, selected_indices]
    
    # 2. Scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_sel)
    
    X_val_scaled = None
    if X_val is not None:
        X_val_sel = X_val.iloc[:, selected_indices]
        X_val_scaled = scaler.transform(X_val_sel)
        
    return scaler, selected_indices, X_train_scaled, X_val_scaled

# =========================
# MODEL TRAINING & EVALUATION
# =========================

def tune_hyperparameters(X_train, y_train, random_state):
    """
    Use RandomizedSearchCV to find best XGBoost hyperparameters.
    """
    logger.info("Tuning hyperparameters...")
    xgb = XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        tree_method="hist",
        n_jobs=-1,
        random_state=random_state
    )
    
    scorer = make_scorer(f1_score)
    
    search = RandomizedSearchCV(
        xgb,
        param_distributions=Config.XGB_PARAM_DIST,
        n_iter=Config.N_ITER_SEARCH,
        scoring=scorer,
        cv=3, # Internal CV for hyperparam tuning
        verbose=1,
        random_state=random_state,
        n_jobs=-1
    )
    
    search.fit(X_train, y_train)
    logger.info(f"Best params: {search.best_params_}")
    logger.info(f"Best internal CV F1: {search.best_score_:.4f}")
    
    return search.best_estimator_, search.best_params_

def find_best_threshold(y_true, y_scores):
    """Find the threshold that maximizes F1 score."""
    thresholds = np.linspace(0.1, 0.9, 81)
    best_thr = 0.5
    best_f1 = -1.0
    
    for thr in thresholds:
        y_pred = (y_scores >= thr).astype(int)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_thr = thr
            
    return best_thr, best_f1

def evaluate_model(model, X_test, y_test, threshold=0.5):
    """Calculate various metrics on test set."""
    y_probs = model.predict_proba(X_test)[:, 1]
    y_pred = (y_probs >= threshold).astype(int)
    
    acc = accuracy_score(y_test, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(y_test, y_pred, average="binary", zero_division=0)
    
    try:
        roc = roc_auc_score(y_test, y_probs)
        pr_auc = average_precision_score(y_test, y_probs)
    except:
        roc, pr_auc = np.nan, np.nan
        
    return {
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "roc_auc": roc,
        "pr_auc": pr_auc
    }

# =========================
# MAIN PIPELINE
# =========================

def run_cross_validation(X, y, groups):
    """Run Stratified Group K-Fold CV."""
    sgkf = StratifiedGroupKFold(n_splits=Config.N_FOLDS)
    
    fold_metrics = []
    
    logger.info(f"Starting {Config.N_FOLDS}-Fold Stratified Group Cross-Validation...")
    
    for fold, (train_idx, val_idx) in enumerate(sgkf.split(X, y, groups)):
        logger.info(f"=== Fold {fold+1}/{Config.N_FOLDS} ===")
        
        X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
        X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]
        
        # Feature Selection & Scaling
        scaler, selected_idx, X_train_scaled, X_val_scaled = select_features_and_scale(
            X_train, y_train, X_val, top_k=Config.TOP_K_FEATURES
        )
        
        # Hyperparameter Tuning (on this fold's training data)
        # To save time in this demo, we could use fixed params, but let's do a quick search
        # or use the best params from a previous run if we wanted to be faster.
        # Here we do a full search per fold to be "production ready" and robust.
        best_model, _ = tune_hyperparameters(X_train_scaled, y_train, Config.RANDOM_STATE)
        
        # Find best threshold on validation set
        y_val_probs = best_model.predict_proba(X_val_scaled)[:, 1]
        best_thr, val_f1 = find_best_threshold(y_val, y_val_probs)
        
        # Evaluate
        metrics = evaluate_model(best_model, X_val_scaled, y_val, best_thr)
        metrics['threshold'] = best_thr
        fold_metrics.append(metrics)
        
        logger.info(f"Fold {fold+1} Metrics: F1={metrics['f1']:.4f}, Acc={metrics['accuracy']:.4f}, Thr={best_thr:.3f}")

    # Aggregate results
    metrics_df = pd.DataFrame(fold_metrics)
    logger.info("\n=== Cross-Validation Results ===")
    logger.info(f"\n{metrics_df.describe().T[['mean', 'std']]}")
    
    return metrics_df.mean().to_dict()

def train_final_model(X, y):
    """Train the final model on the entire dataset."""
    logger.info("Training final model on full dataset...")
    
    # Feature Selection & Scaling on full data
    scaler, selected_idx, X_scaled, _ = select_features_and_scale(X, y, top_k=Config.TOP_K_FEATURES)
    
    # Hyperparameter Tuning on full data
    best_model, best_params = tune_hyperparameters(X_scaled, y, Config.RANDOM_STATE)
    
    # We don't have a validation set to tune threshold here, so we use the average threshold from CV
    # Or we could do a nested split, but for simplicity/production, we often use 0.5 or a conservative value.
    # A better approach: use the threshold that gave best F1 in CV (averaged).
    
    return best_model, scaler, selected_idx, best_params

def save_artifacts(model, scaler, feature_names, selected_indices, metrics, config, avg_threshold):
    """Save model and metadata."""
    Config.MODEL_DIR.mkdir(parents=True, exist_ok=True)
    
    model_path = Config.MODEL_DIR / "xgb_ecg_seizure_model.joblib"
    scaler_path = Config.MODEL_DIR / "scaler.joblib"
    info_path = Config.MODEL_DIR / "model_info.json"
    
    dump(model, model_path)
    dump(scaler, scaler_path)
    
    # Get names of selected features
    selected_names = [feature_names[i] for i in selected_indices]
    
    info = {
        "cv_metrics_mean": metrics,
        "final_threshold_estimated": avg_threshold,
        "selected_feature_indices": selected_indices.tolist(),
        "feature_names": selected_names,
        "config": {
            "TOP_K_FEATURES": config.TOP_K_FEATURES,
            "N_FOLDS": config.N_FOLDS,
            "RANDOM_STATE": config.RANDOM_STATE,
        },
        "timestamp": datetime.now().isoformat()
    }
    
    with open(info_path, "w", encoding="utf-8") as f:
        json.dump(info, f, indent=2)
        
    logger.info(f"Artifacts saved to {Config.MODEL_DIR}")

def main():
    try:
        # 1. Load Data
        ecg_df = load_ecg_data(Config.CSV_PATH)
        
        # 2. Preprocess
        X, y, groups = preprocess_data(ecg_df)
        feature_names = list(X.columns)
        
        # 3. Cross-Validation
        avg_metrics = run_cross_validation(X, y, groups)
        avg_threshold = avg_metrics['threshold']
        
        # 4. Final Training
        final_model, scaler, selected_idx, best_params = train_final_model(X, y)
        
        # 5. Save Artifacts
        save_artifacts(
            final_model, 
            scaler, 
            feature_names, 
            selected_idx, 
            avg_metrics, 
            Config, 
            avg_threshold
        )
        
        logger.info("Pipeline completed successfully.")
        
    except Exception as e:
        logger.critical(f"Pipeline failed: {e}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main()
