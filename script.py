import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, KBinsDiscretizer
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    roc_auc_score,
    average_precision_score,
)
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, IsolationForest
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC, OneClassSVM
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neural_network import MLPClassifier
from sklearn.feature_selection import mutual_info_classif

from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

# ======================================================================
# 1. Load CSV
# ======================================================================

# Change path if needed
df = pd.read_csv("ml_ready_balanced.csv")

# ======================================================================
# 2. Keep only ECG modality
# ======================================================================

ecg_df = df[df["Modality"] == "ECG"].copy()

print(f"Total rows (ECG only): {len(ecg_df)}")
print("Label distribution (ECG only):")
print(ecg_df["Label"].value_counts())

# ======================================================================
# 3. Subject-wise split using existing 'split' column
#    - 'train' subjects -> we further split into train + val (by subject)
#    - 'test' subjects stay as test
# ======================================================================

train_subjects = ecg_df.loc[ecg_df["split"] == "train", "Subject_ID"].unique()
test_subjects = ecg_df.loc[ecg_df["split"] == "test", "Subject_ID"].unique()

train_subj, val_subj = train_test_split(
    train_subjects,
    test_size=0.2,
    random_state=RANDOM_STATE,
    shuffle=True,
)

train_df = ecg_df[ecg_df["Subject_ID"].isin(train_subj)].copy()
val_df   = ecg_df[ecg_df["Subject_ID"].isin(val_subj)].copy()
test_df  = ecg_df[ecg_df["Subject_ID"].isin(test_subjects)].copy()

print("\nSubjects per split:")
print("  Train subjects:", len(np.unique(train_df["Subject_ID"])))
print("  Val subjects:  ", len(np.unique(val_df["Subject_ID"])))
print("  Test subjects: ", len(np.unique(test_df["Subject_ID"])))

# ======================================================================
# 4. Drop metadata columns (to avoid leakage / cheating)
# ======================================================================

cols_to_drop = [
    "Subject_ID",
    "Modality",
    "split",
    "Onset_Seconds",
    "Duration_Seconds",
    "Epoch_Index",
]

for col in cols_to_drop:
    if col in train_df.columns:
        train_df.drop(columns=[col], inplace=True)
        val_df.drop(columns=[col], inplace=True)
        test_df.drop(columns=[col], inplace=True)

# ======================================================================
# 5. Split features / labels
# ======================================================================

y_train = train_df["Label"].values
y_val   = val_df["Label"].values
y_test  = test_df["Label"].values

X_train = train_df.drop(columns=["Label"])
X_val   = val_df.drop(columns=["Label"])
X_test  = test_df.drop(columns=["Label"])

# ======================================================================
# 6. One-hot encode categorical features (Sex) using train as reference
# ======================================================================

categorical_cols = []
if "Sex" in X_train.columns:
    categorical_cols.append("Sex")

X_train_enc = pd.get_dummies(X_train, columns=categorical_cols, drop_first=True)
X_val_enc   = pd.get_dummies(X_val,   columns=categorical_cols, drop_first=True)
X_test_enc  = pd.get_dummies(X_test,  columns=categorical_cols, drop_first=True)

X_val_enc  = X_val_enc.reindex(columns=X_train_enc.columns, fill_value=0)
X_test_enc = X_test_enc.reindex(columns=X_train_enc.columns, fill_value=0)

print("\nFeature dimension after encoding:", X_train_enc.shape[1])

# ======================================================================
# 7. Scale features (StandardScaler on train only)
# ======================================================================

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_enc)
X_val_scaled   = scaler.transform(X_val_enc)
X_test_scaled  = scaler.transform(X_test_enc)

# ======================================================================
# 8. Feature selection (mutual information, top-K)
#    This mimics HRV papers where they carefully select features.
# ======================================================================

def select_top_k_features(X_train_arr, y_train_arr, X_val_arr, X_test_arr, k=200):
    mi = mutual_info_classif(X_train_arr, y_train_arr, random_state=RANDOM_STATE)
    idx_sorted = np.argsort(mi)[::-1]  # descending
    k = min(k, X_train_arr.shape[1])
    top_idx = idx_sorted[:k]
    return (
        X_train_arr[:, top_idx],
        X_val_arr[:, top_idx],
        X_test_arr[:, top_idx],
        top_idx,
        mi[top_idx],
    )

X_train_sel, X_val_sel, X_test_sel, selected_idx, selected_mi = select_top_k_features(
    X_train_scaled, y_train, X_val_scaled, X_test_scaled, k=200
)

print(f"\nSelected top {X_train_sel.shape[1]} features by mutual information.")

# ======================================================================
# 9. Helper: generic evaluation function
# ======================================================================

def evaluate_model(name, model, X_train_arr, y_train_arr, X_test_arr, y_test_arr):
    """
    Fit model on X_train_arr/y_train_arr and evaluate on X_test_arr/y_test_arr.
    Returns a dict with metrics.
    """
    print(f"\n[INFO] Training {name}...")
    model.fit(X_train_arr, y_train_arr)

    y_pred = model.predict(X_test_arr)

    acc = accuracy_score(y_test_arr, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(
        y_test_arr, y_pred, average="binary", zero_division=0
    )

    y_scores = None
    if hasattr(model, "predict_proba"):
        try:
            y_scores = model.predict_proba(X_test_arr)[:, 1]
        except Exception:
            y_scores = None
    elif hasattr(model, "decision_function"):
        try:
            y_scores = model.decision_function(X_test_arr)
        except Exception:
            y_scores = None

    if y_scores is not None:
        try:
            roc = roc_auc_score(y_test_arr, y_scores)
            pr_auc = average_precision_score(y_test_arr, y_scores)
        except Exception:
            roc = np.nan
            pr_auc = np.nan
    else:
        roc = np.nan
        pr_auc = np.nan

    print(
        f"{name} -> "
        f"Acc: {acc:.3f}, Prec: {prec:.3f}, Rec: {rec:.3f}, "
        f"F1: {f1:.3f}, ROC-AUC: {roc:.3f}, PR-AUC: {pr_auc:.3f}"
    )

    return {
        "model": name,
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "roc_auc": roc,
        "pr_auc": pr_auc,
    }


results = []

# ======================================================================
# 10. Supervised models on selected features (global cross-subject)
# ======================================================================

# LightGBM
lgbm = LGBMClassifier(
    n_estimators=500,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=RANDOM_STATE,
    n_jobs=-1,
)
results.append(
    evaluate_model("LightGBM", lgbm, X_train_sel, y_train, X_test_sel, y_test)
)

# XGBoost
xgb = XGBClassifier(
    n_estimators=500,
    learning_rate=0.05,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    objective="binary:logistic",
    eval_metric="logloss",
    n_jobs=-1,
    tree_method="hist",
    random_state=RANDOM_STATE,
)
results.append(
    evaluate_model("XGBoost", xgb, X_train_sel, y_train, X_test_sel, y_test)
)

# CatBoost
cat = CatBoostClassifier(
    depth=6,
    learning_rate=0.05,
    loss_function="Logloss",
    eval_metric="AUC",
    random_state=RANDOM_STATE,
    verbose=False,
)
results.append(
    evaluate_model("CatBoost", cat, X_train_sel, y_train, X_test_sel, y_test)
)

# ExtraTrees
et = ExtraTreesClassifier(
    n_estimators=400,
    max_depth=None,
    n_jobs=-1,
    random_state=RANDOM_STATE,
)
results.append(
    evaluate_model("ExtraTrees", et, X_train_sel, y_train, X_test_sel, y_test)
)

# RandomForest
rf = RandomForestClassifier(
    n_estimators=400,
    max_depth=None,
    n_jobs=-1,
    random_state=RANDOM_STATE,
)
results.append(
    evaluate_model("RandomForest", rf, X_train_sel, y_train, X_test_sel, y_test)
)

# Logistic Regression with ElasticNet
logreg = LogisticRegression(
    penalty="elasticnet",
    solver="saga",
    l1_ratio=0.5,
    max_iter=500,
    n_jobs=-1,
    random_state=RANDOM_STATE,
)
results.append(
    evaluate_model(
        "LogisticRegression_ElasticNet",
        logreg,
        X_train_sel,
        y_train,
        X_test_sel,
        y_test,
    )
)

# Linear SVM
linsvm = LinearSVC(C=1.0, random_state=RANDOM_STATE)
results.append(
    evaluate_model("LinearSVC", linsvm, X_train_sel, y_train, X_test_sel, y_test)
)

# LDA with shrinkage
lda = LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")
results.append(
    evaluate_model("LDA_shrinkage", lda, X_train_sel, y_train, X_test_sel, y_test)
)

# MLP
mlp = MLPClassifier(
    hidden_layer_sizes=(512, 256, 64),
    activation="relu",
    solver="adam",
    alpha=1e-4,
    batch_size=256,
    learning_rate_init=1e-3,
    max_iter=50,
    early_stopping=True,
    n_iter_no_change=5,
    validation_fraction=0.1,
    random_state=RANDOM_STATE,
)
results.append(
    evaluate_model("MLP", mlp, X_train_sel, y_train, X_test_sel, y_test)
)

# ======================================================================
# 11. Anomaly detection branch (only normal windows in train)
#     Inspired by HRV anomaly detection approaches.
# ======================================================================

def evaluate_anomaly_model(name, model, X_train_arr, y_train_arr, X_test_arr, y_test_arr):
    """
    Train an anomaly detector on normal (label 0) train samples,
    then evaluate mapping anomalies -> seizure (1).
    """
    print(f"\n[INFO] Training anomaly detector {name} on normal train data...")
    X_train_norm = X_train_arr[y_train_arr == 0]
    model.fit(X_train_norm)

    pred_raw = model.predict(X_test_arr)
    # For OneClassSVM and IsolationForest:
    #   +1 = inlier (normal), -1 = outlier (anomaly)
    y_pred = np.where(pred_raw == -1, 1, 0)

    acc = accuracy_score(y_test_arr, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(
        y_test_arr, y_pred, average="binary", zero_division=0
    )

    # No probabilities typically available -> ROC/PR AUC left as NaN
    roc = np.nan
    pr_auc = np.nan

    print(
        f"{name} (anomaly) -> "
        f"Acc: {acc:.3f}, Prec: {prec:.3f}, Rec: {rec:.3f}, F1: {f1:.3f}"
    )

    return {
        "model": name,
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "roc_auc": roc,
        "pr_auc": pr_auc,
    }

# One-Class SVM
oc_svm = OneClassSVM(
    kernel="rbf",
    nu=0.1,        # fraction of anomalies expected
    gamma="scale",
)
results.append(
    evaluate_anomaly_model(
        "OneClassSVM_anomaly", oc_svm, X_train_sel, y_train, X_test_sel, y_test
    )
)

# Isolation Forest
iso_forest = IsolationForest(
    n_estimators=300,
    contamination=0.1,
    random_state=RANDOM_STATE,
    n_jobs=-1,
)
results.append(
    evaluate_anomaly_model(
        "IsolationForest_anomaly", iso_forest, X_train_sel, y_train, X_test_sel, y_test
    )
)

# ======================================================================
# 12. XCS-style discretized data + placeholders for XCS/UCS/ExSTraCS/XCSF
# ======================================================================

def prepare_discretized_features(X_train_arr, X_val_arr, X_test_arr, n_bins=3):
    kbd = KBinsDiscretizer(
        n_bins=n_bins,
        encode="ordinal",
        strategy="quantile",
    )
    X_train_disc = kbd.fit_transform(X_train_arr)
    X_val_disc   = kbd.transform(X_val_arr)
    X_test_disc  = kbd.transform(X_test_arr)
    return X_train_disc, X_val_disc, X_test_disc, kbd

X_train_disc, X_val_disc, X_test_disc, disc_transformer = prepare_discretized_features(
    X_train_sel, X_val_sel, X_test_sel, n_bins=3
)

def evaluate_xcs_placeholder(name):
    """
    Placeholder for XCS/UCS/ExSTraCS/XCSF.
    Replace with real implementation (fit/predict) when available.
    """
    print(f"[INFO] {name} not evaluated: external XCS implementation required.")
    return {
        "model": name,
        "accuracy": np.nan,
        "precision": np.nan,
        "recall": np.nan,
        "f1": np.nan,
        "roc_auc": np.nan,
        "pr_auc": np.nan,
    }

results.append(evaluate_xcs_placeholder("XCS_discretized"))
results.append(evaluate_xcs_placeholder("UCS_supervised"))
results.append(evaluate_xcs_placeholder("ExSTraCS"))
results.append(evaluate_xcs_placeholder("XCSF_continuous"))

# ======================================================================
# 13. Patient-specific evaluation (LightGBM) - average across subjects
#     This mimics "patient-specific" literature setups.
# ======================================================================

def patient_specific_evaluation(base_model, X_all_df, y_all_arr, subject_ids_arr):
    """
    For each subject, split their data into train/test, train a model on that subject only,
    and compute metrics. Return mean metrics across subjects.
    """
    subjects = np.unique(subject_ids_arr)
    subject_metrics = []

    for subj in subjects:
        mask = subject_ids_arr == subj
        X_subj = X_all_df[mask]
        y_subj = y_all_arr[mask]

        # skip if too few samples or too few seizure events
        if len(np.unique(y_subj)) < 2 or len(y_subj) < 20:
            continue

        X_tr_s, X_te_s, y_tr_s, y_te_s = train_test_split(
            X_subj,
            y_subj,
            test_size=0.3,
            random_state=RANDOM_STATE,
            stratify=y_subj,
        )

        model = base_model
        model.fit(X_tr_s, y_tr_s)
        y_pred_s = model.predict(X_te_s)

        acc_s = accuracy_score(y_te_s, y_pred_s)
        prec_s, rec_s, f1_s, _ = precision_recall_fscore_support(
            y_te_s, y_pred_s, average="binary", zero_division=0
        )

        subject_metrics.append((acc_s, prec_s, rec_s, f1_s))

    if not subject_metrics:
        return None

    subject_metrics = np.array(subject_metrics)
    mean_acc, mean_prec, mean_rec, mean_f1 = subject_metrics.mean(axis=0)

    return {
        "accuracy": mean_acc,
        "precision": mean_prec,
        "recall": mean_rec,
        "f1": mean_f1,
    }

# For patient-specific, we need original ECG-all arrays (not subject-dropped).
# We can reuse ecg_df but reprocess quickly.
ecg_ps = df[df["Modality"] == "ECG"].copy()
subj_ids_ps = ecg_ps["Subject_ID"].values
labels_ps = ecg_ps["Label"].values

# Drop same metadata, then encode + scale using full ECG set for patient-specific
for col in cols_to_drop:
    if col in ecg_ps.columns:
        ecg_ps.drop(columns=[col], inplace=True)

X_ps = ecg_ps.drop(columns=["Label"])
y_ps = labels_ps

X_ps_enc = pd.get_dummies(X_ps, columns=[c for c in ["Sex"] if c in X_ps.columns], drop_first=True)
# Scale with new scaler (patient-specific analysis is independent branch)
scaler_ps = StandardScaler()
X_ps_scaled = scaler_ps.fit_transform(X_ps_enc)

# Feature selection for patient-specific branch (keep same K)
X_ps_sel = X_ps_scaled[:, selected_idx] if X_ps_scaled.shape[1] >= len(selected_idx) else X_ps_scaled

lgbm_ps = LGBMClassifier(
    n_estimators=300,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=RANDOM_STATE,
    n_jobs=-1,
)

ps_metrics = patient_specific_evaluation(
    lgbm_ps,
    X_ps_sel,
    y_ps,
    subj_ids_ps,
)

if ps_metrics is not None:
    print("\n[INFO] Patient-specific LightGBM (mean across subjects):")
    print(
        f"Acc: {ps_metrics['accuracy']:.3f}, "
        f"Prec: {ps_metrics['precision']:.3f}, "
        f"Rec: {ps_metrics['recall']:.3f}, "
        f"F1: {ps_metrics['f1']:.3f}"
    )

    results.append(
        {
            "model": "LightGBM_patient_specific_mean",
            "accuracy": ps_metrics["accuracy"],
            "precision": ps_metrics["precision"],
            "recall": ps_metrics["recall"],
            "f1": ps_metrics["f1"],
            "roc_auc": np.nan,
            "pr_auc": np.nan,
        }
    )

# ======================================================================
# 14. Final comparison table (cross-subject + anomaly + placeholders + PS)
# ======================================================================

results_df = pd.DataFrame(results)
results_df = results_df.sort_values(by="f1", ascending=False)

print("\n================= FINAL COMPARISON (TEST SET & OTHERS) =================")
print(results_df.to_string(index=False))
