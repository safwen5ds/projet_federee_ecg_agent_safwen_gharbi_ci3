import warnings
warnings.filterwarnings("ignore")

import time
import numpy as np
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, chi2, f_classif
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    roc_auc_score,
    average_precision_score,
)
from sklearn.base import BaseEstimator, TransformerMixin


RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)


class NonNegativeTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, epsilon=1e-10):
        self.epsilon = epsilon
        self.min_ = None

    def fit(self, X, y=None):
        self.min_ = np.min(X, axis=0)
        return self

    def transform(self, X):
        return X - self.min_ + self.epsilon


def evaluate_model_on_test(name, model, X_test, y_test):
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(
        y_test, y_pred, average="binary", zero_division=0
    )

    y_scores = None
    if hasattr(model, "predict_proba"):
        try:
            y_scores = model.predict_proba(X_test)[:, 1]
        except Exception:
            y_scores = None
    elif hasattr(model, "decision_function"):
        try:
            y_scores = model.decision_function(X_test)
        except Exception:
            y_scores = None

    if y_scores is not None:
        try:
            roc = roc_auc_score(y_test, y_scores)
            pr_auc = average_precision_score(y_test, y_scores)
        except Exception:
            roc = np.nan
            pr_auc = np.nan
    else:
        roc = np.nan
        pr_auc = np.nan

    print(f"\n=== Test metrics for {name} ===")
    print(f"Accuracy : {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall   : {rec:.4f}")
    print(f"F1-score : {f1:.4f}")
    print(f"ROC AUC  : {roc:.4f}")
    print(f"PR  AUC  : {pr_auc:.4f}")

    return {
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "roc_auc": roc,
        "pr_auc": pr_auc,
    }


print("=" * 80)
print("GRID SEARCH: Chi-Square + MLP  &  ANOVA + GradientBoosting")
print("=" * 80)

df = pd.read_csv("ml_ready_balanced.csv")

ecg_df = df[df["Modality"] == "ECG"].copy()
print(f"\nTotal rows (ECG only): {len(ecg_df)}")
print("Label distribution (ECG only):")
print(ecg_df["Label"].value_counts())

train_subjects = ecg_df.loc[ecg_df["split"] == "train", "Subject_ID"].unique()
test_subjects = ecg_df.loc[ecg_df["split"] == "test", "Subject_ID"].unique()

train_subj, val_subj = train_test_split(
    train_subjects,
    test_size=0.2,
    random_state=RANDOM_STATE,
    shuffle=True,
)

train_df = ecg_df[ecg_df["Subject_ID"].isin(train_subj)].copy()
val_df = ecg_df[ecg_df["Subject_ID"].isin(val_subj)].copy()
test_df = ecg_df[ecg_df["Subject_ID"].isin(test_subjects)].copy()

print("\nSubjects per split:")
print(f"  Train subjects: {train_df['Subject_ID'].nunique()}")
print(f"  Val subjects  : {val_df['Subject_ID'].nunique()}")
print(f"  Test subjects : {test_df['Subject_ID'].nunique()}")

cols_to_drop = [
    "Subject_ID",
    "Modality",
    "split",
    "Onset_Seconds",
    "Duration_Seconds",
    "Epoch_Index",
]
existing_to_drop = [c for c in cols_to_drop if c in train_df.columns]
train_df.drop(columns=existing_to_drop, inplace=True)
val_df.drop(columns=existing_to_drop, inplace=True)
test_df.drop(columns=existing_to_drop, inplace=True)

y_train = train_df["Label"].values
y_val = val_df["Label"].values
y_test = test_df["Label"].values

X_train = train_df.drop(columns=["Label"])
X_val = val_df.drop(columns=["Label"])
X_test = test_df.drop(columns=["Label"])

categorical_cols = []
if "Sex" in X_train.columns:
    categorical_cols.append("Sex")

X_train_enc = pd.get_dummies(X_train, columns=categorical_cols, drop_first=True)
X_val_enc = pd.get_dummies(X_val, columns=categorical_cols, drop_first=True)
X_test_enc = pd.get_dummies(X_test, columns=categorical_cols, drop_first=True)

X_val_enc = X_val_enc.reindex(columns=X_train_enc.columns, fill_value=0)
X_test_enc = X_test_enc.reindex(columns=X_train_enc.columns, fill_value=0)

print(f"\nFinal input feature dimension (after encoding): {X_train_enc.shape[1]}")

X_train_np = X_train_enc.values
X_test_np = X_test_enc.values

cv = StratifiedKFold(
    n_splits=3,
    shuffle=True,
    random_state=RANDOM_STATE,
)

print("\n" + "=" * 80)
print("GRID SEARCH: Chi-Square + MLPClassifier")
print("=" * 80)

pipeline_mlp = Pipeline(
    steps=[
        ("nonneg", NonNegativeTransformer()),
        ("select", SelectKBest(score_func=chi2)),
        ("scaler", StandardScaler(with_mean=False)),
        ("clf", MLPClassifier(random_state=RANDOM_STATE)),
    ]
)

param_grid_mlp = {
    "select__k": [50, 100, 150, "all"],
    "clf__hidden_layer_sizes": [(50,), (100,), (100, 50)],
    "clf__alpha": [1e-4, 1e-3, 1e-2],
    "clf__learning_rate_init": [0.001, 0.01],
    "clf__max_iter": [300],
}

grid_mlp = GridSearchCV(
    estimator=pipeline_mlp,
    param_grid=param_grid_mlp,
    scoring="f1",
    cv=cv,
    n_jobs=-1,
    verbose=2,
)

t0 = time.time()
grid_mlp.fit(X_train_np, y_train)
mlp_search_time = time.time() - t0

print(f"\nBest params (MLP + Chi2): {grid_mlp.best_params_}")
print(f"Best CV F1 (MLP + Chi2): {grid_mlp.best_score_:.4f}")
print(f"Grid search time (MLP + Chi2): {mlp_search_time:.2f} s")

best_mlp_model = grid_mlp.best_estimator_

metrics_mlp = evaluate_model_on_test("MLP + Chi2 (best grid)", best_mlp_model, X_test_np, y_test)

print("\n" + "=" * 80)
print("GRID SEARCH: ANOVA F-test + GradientBoostingClassifier")
print("=" * 80)

pipeline_gb = Pipeline(
    steps=[
        ("scaler", StandardScaler()),
        ("select", SelectKBest(score_func=f_classif)),
        ("clf", GradientBoostingClassifier(random_state=RANDOM_STATE)),
    ]
)

param_grid_gb = {
    "select__k": [50, 100, 150, "all"],
    "clf__n_estimators": [100, 200],
    "clf__learning_rate": [0.05, 0.1],
    "clf__max_depth": [3, 5],
}

grid_gb = GridSearchCV(
    estimator=pipeline_gb,
    param_grid=param_grid_gb,
    scoring="f1",
    cv=cv,
    n_jobs=-1,
    verbose=2,
)

t0 = time.time()
grid_gb.fit(X_train_np, y_train)
gb_search_time = time.time() - t0

print(f"\nBest params (GB + ANOVA): {grid_gb.best_params_}")
print(f"Best CV F1 (GB + ANOVA): {grid_gb.best_score_:.4f}")
print(f"Grid search time (GB + ANOVA): {gb_search_time:.2f} s")

best_gb_model = grid_gb.best_estimator_

metrics_gb = evaluate_model_on_test("GradientBoosting + ANOVA (best grid)", best_gb_model, X_test_np, y_test)

print("\n" + "=" * 80)
print("SELECTING GLOBAL BEST MODEL ON TEST SET")
print("=" * 80)

f1_mlp = metrics_mlp["f1"]
f1_gb = metrics_gb["f1"]

if f1_mlp >= f1_gb:
    best_model_name = "MLP + Chi2 (grid best)"
    best_model = best_mlp_model
    best_metrics = metrics_mlp
else:
    best_model_name = "GradientBoosting + ANOVA (grid best)"
    best_model = best_gb_model
    best_metrics = metrics_gb

print(f"\nGlobal best model on TEST (by F1): {best_model_name}")
print(f"F1-score: {best_metrics['f1']:.4f}")
print(f"Accuracy: {best_metrics['accuracy']:.4f}")
print(f"ROC AUC : {best_metrics['roc_auc']:.4f}")

save_path = "best_ecg_model_grid.joblib"
joblib.dump(best_model, save_path)
print(f"\n[INFO] Best model saved to: {save_path}")

print("\n" + "=" * 80)
print("GRID SEARCH EXPERIMENT COMPLETE")
print("=" * 80)
