import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import time

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    roc_auc_score,
    average_precision_score,
)

from sklearn.feature_selection import (
    mutual_info_classif,
    chi2,
    f_classif,
    RFE,
    SelectKBest,
    VarianceThreshold,
    SelectFromModel,
)
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

try:
    from sklearn.utils.discovery import all_estimators
except ImportError:
    from sklearn.utils import all_estimators

try:
    from lightgbm import LGBMClassifier
    HAS_LIGHTGBM = True
except ImportError:
    HAS_LIGHTGBM = False

try:
    from xgboost import XGBClassifier
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False

try:
    from catboost import CatBoostClassifier
    HAS_CATBOOST = True
except ImportError:
    HAS_CATBOOST = False

RANDOM_STATE = 42
N_JOBS = -1
np.random.seed(RANDOM_STATE)

print("=" * 80)
print("FEATURE SELECTION + *ALL* CLASSIFIERS COMPARISON")
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
print(f"  Val subjects:   {val_df['Subject_ID'].nunique()}")
print(f"  Test subjects:  {test_df['Subject_ID'].nunique()}")

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

print(f"\nOriginal feature dimension: {X_train_enc.shape[1]}")

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_enc)
X_val_scaled = scaler.transform(X_val_enc)
X_test_scaled = scaler.transform(X_test_enc)

K_FEATURES = 100


def apply_no_selection(X_train, X_val, X_test, y_train, k=K_FEATURES):
    idx = list(range(X_train.shape[1]))
    return X_train, X_val, X_test, idx


def apply_mutual_information(X_train, X_val, X_test, y_train, k=K_FEATURES):
    mi = mutual_info_classif(X_train, y_train, random_state=RANDOM_STATE)
    idx_sorted = np.argsort(mi)[::-1]
    k = min(k, X_train.shape[1])
    top_idx = idx_sorted[:k]
    return X_train[:, top_idx], X_val[:, top_idx], X_test[:, top_idx], top_idx


def apply_chi_square(X_train, X_val, X_test, y_train, k=K_FEATURES):
    global_min = X_train.min()
    X_train_nonneg = X_train - global_min + 1e-10
    X_val_nonneg = X_val - global_min + 1e-10
    X_test_nonneg = X_test - global_min + 1e-10
    selector = SelectKBest(score_func=chi2, k=min(k, X_train.shape[1]))
    X_train_sel = selector.fit_transform(X_train_nonneg, y_train)
    X_val_sel = selector.transform(X_val_nonneg)
    X_test_sel = selector.transform(X_test_nonneg)
    top_idx = selector.get_support(indices=True)
    return X_train_sel, X_val_sel, X_test_sel, top_idx


def apply_anova_f_test(X_train, X_val, X_test, y_train, k=K_FEATURES):
    selector = SelectKBest(score_func=f_classif, k=min(k, X_train.shape[1]))
    X_train_sel = selector.fit_transform(X_train, y_train)
    X_val_sel = selector.transform(X_val)
    X_test_sel = selector.transform(X_test)
    top_idx = selector.get_support(indices=True)
    return X_train_sel, X_val_sel, X_test_sel, top_idx


def apply_rfe(X_train, X_val, X_test, y_train, k=K_FEATURES):
    base_estimator = LogisticRegression(
        max_iter=200,
        random_state=RANDOM_STATE,
        n_jobs=N_JOBS,
    )
    n_features_to_select = min(k, X_train.shape[1])
    selector = RFE(
        estimator=base_estimator,
        n_features_to_select=n_features_to_select,
        step=0.2,
    )
    X_train_sel = selector.fit_transform(X_train, y_train)
    X_val_sel = selector.transform(X_val)
    X_test_sel = selector.transform(X_test)
    top_idx = selector.get_support(indices=True)
    return X_train_sel, X_val_sel, X_test_sel, top_idx


def apply_l1_lasso(X_train, X_val, X_test, y_train, k=K_FEATURES):
    l1_logreg = LogisticRegression(
        penalty="l1",
        solver="liblinear",
        C=0.1,
        random_state=RANDOM_STATE,
        max_iter=500,
    )
    selector = SelectFromModel(
        l1_logreg,
        prefit=False,
        max_features=min(k, X_train.shape[1]),
    )
    X_train_sel = selector.fit_transform(X_train, y_train)
    X_val_sel = selector.transform(X_val)
    X_test_sel = selector.transform(X_test)
    top_idx = selector.get_support(indices=True)
    return X_train_sel, X_val_sel, X_test_sel, top_idx


def apply_tree_based(X_train, X_val, X_test, y_train, k=K_FEATURES):
    rf = RandomForestClassifier(
        n_estimators=200,
        random_state=RANDOM_STATE,
        n_jobs=N_JOBS,
        max_depth=10,
    )
    selector = SelectFromModel(
        rf,
        prefit=False,
        max_features=min(k, X_train.shape[1]),
    )
    X_train_sel = selector.fit_transform(X_train, y_train)
    X_val_sel = selector.transform(X_val)
    X_test_sel = selector.transform(X_test)
    top_idx = selector.get_support(indices=True)
    return X_train_sel, X_val_sel, X_test_sel, top_idx


def apply_variance_threshold(X_train, X_val, X_test, y_train, k=K_FEATURES):
    selector = VarianceThreshold(threshold=0.01)
    X_train_var = selector.fit_transform(X_train)
    X_val_var = selector.transform(X_val)
    X_test_var = selector.transform(X_test)
    variances = np.var(X_train_var, axis=0)
    idx_sorted = np.argsort(variances)[::-1]
    k = min(k, X_train_var.shape[1])
    top_idx_var = idx_sorted[:k]
    original_idx = selector.get_support(indices=True)
    top_idx = original_idx[top_idx_var]
    return (
        X_train_var[:, top_idx_var],
        X_val_var[:, top_idx_var],
        X_test_var[:, top_idx_var],
        top_idx,
    )


def apply_correlation(X_train, X_val, X_test, y_train, k=K_FEATURES):
    correlations = []
    for i in range(X_train.shape[1]):
        c = np.corrcoef(X_train[:, i], y_train)[0, 1]
        if np.isnan(c):
            c = 0.0
        correlations.append(abs(c))
    correlations = np.array(correlations)
    idx_sorted = np.argsort(correlations)[::-1]
    k = min(k, X_train.shape[1])
    top_idx = idx_sorted[:k]
    return X_train[:, top_idx], X_val[:, top_idx], X_test[:, top_idx], top_idx


def apply_pca(X_train, X_val, X_test, y_train, k=K_FEATURES):
    n_components = min(k, X_train.shape[1])
    pca = PCA(n_components=n_components, random_state=RANDOM_STATE)
    X_train_pca = pca.fit_transform(X_train)
    X_val_pca = pca.transform(X_val)
    X_test_pca = pca.transform(X_test)
    top_idx = list(range(pca.n_components_))
    return X_train_pca, X_val_pca, X_test_pca, top_idx


def apply_random_selection(X_train, X_val, X_test, y_train, k=K_FEATURES):
    np.random.seed(RANDOM_STATE)
    k = min(k, X_train.shape[1])
    top_idx = np.random.choice(X_train.shape[1], k, replace=False)
    return X_train[:, top_idx], X_val[:, top_idx], X_test[:, top_idx], top_idx


def apply_lgbm_importance(X_train, X_val, X_test, y_train, k=K_FEATURES):
    if not HAS_LIGHTGBM:
        raise RuntimeError("LightGBM not available")
    lgbm = LGBMClassifier(
        n_estimators=300,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=RANDOM_STATE,
        n_jobs=N_JOBS,
        verbose=-1,
    )
    lgbm.fit(X_train, y_train)
    importances = lgbm.feature_importances_
    idx_sorted = np.argsort(importances)[::-1]
    k = min(k, X_train.shape[1])
    top_idx = idx_sorted[:k]
    return (
        X_train[:, top_idx],
        X_val[:, top_idx],
        X_test[:, top_idx],
        top_idx,
    )


FEATURE_SELECTION_METHODS = {
    "No_Selection": apply_no_selection,
    "Mutual_Information": apply_mutual_information,
    "Chi_Square": apply_chi_square,
    "ANOVA_F_Test": apply_anova_f_test,
    "RFE_LogReg": apply_rfe,
    "L1_Lasso": apply_l1_lasso,
    "Tree_RF_Importance": apply_tree_based,
    "Variance_Threshold": apply_variance_threshold,
    "Pearson_Correlation": apply_correlation,
    "PCA": apply_pca,
    "Random_Selection": apply_random_selection,
}

if HAS_LIGHTGBM:
    FEATURE_SELECTION_METHODS["LGBM_Importance"] = apply_lgbm_importance


def evaluate_model(name, model, X_train_arr, y_train_arr, X_test_arr, y_test_arr, verbose=False):
    if verbose:
        print(f"  Training {name}...", end=" ")
    t0 = time.time()
    model.fit(X_train_arr, y_train_arr)
    train_time = time.time() - t0
    t0 = time.time()
    y_pred = model.predict(X_test_arr)
    predict_time = time.time() - t0
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
    if verbose:
        print(f"F1: {f1:.3f}, ROC-AUC: {roc:.3f}")
    return {
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "roc_auc": roc,
        "pr_auc": pr_auc,
        "train_time": train_time,
        "predict_time": predict_time,
    }


def get_all_sklearn_classifiers():
    classifiers = {}
    for name, Cls in all_estimators(type_filter="classifier"):
        try:
            clf = Cls()
        except TypeError:
            continue
        except Exception:
            continue
        classifiers[name] = clf
    if HAS_LIGHTGBM:
        classifiers["LGBMClassifier_Custom"] = LGBMClassifier(
            n_estimators=300,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=RANDOM_STATE,
            n_jobs=N_JOBS,
            verbose=-1,
        )
    if HAS_XGBOOST:
        classifiers["XGBClassifier_Custom"] = XGBClassifier(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=RANDOM_STATE,
            n_jobs=N_JOBS,
            eval_metric="logloss",
        )
    if HAS_CATBOOST:
        classifiers["CatBoostClassifier_Custom"] = CatBoostClassifier(
            iterations=300,
            learning_rate=0.05,
            depth=6,
            loss_function="Logloss",
            verbose=False,
            random_seed=RANDOM_STATE,
        )
    print(f"\nDiscovered {len(classifiers)} classifiers (sklearn + optional external).")
    return classifiers


print("\n" + "=" * 80)
print("RUNNING FEATURE SELECTION + ALL CLASSIFIERS")
print("=" * 80)

all_results = []

for fs_name, fs_method in FEATURE_SELECTION_METHODS.items():
    print(f"\n[{fs_name}]")
    print("-" * 60)
    try:
        t0 = time.time()
        X_tr_sel, X_val_sel, X_te_sel, selected_idx = fs_method(
            X_train_scaled, X_val_scaled, X_test_scaled, y_train
        )
        fs_time = time.time() - t0
        n_features = X_tr_sel.shape[1]
        print(f"  Features selected: {n_features} (FS time: {fs_time:.2f}s)")
        models = get_all_sklearn_classifiers()
        for model_name, model in models.items():
            try:
                metrics = evaluate_model(
                    model_name,
                    model,
                    X_tr_sel,
                    y_train,
                    X_te_sel,
                    y_test,
                    verbose=True,
                )
                all_results.append({
                    "Feature_Selection": fs_name,
                    "Model": model_name,
                    "N_Features": n_features,
                    "FS_Time_s": fs_time,
                    "Train_Time_s": metrics["train_time"],
                    "Predict_Time_s": metrics["predict_time"],
                    "Accuracy": metrics["accuracy"],
                    "Precision": metrics["precision"],
                    "Recall": metrics["recall"],
                    "F1_Score": metrics["f1"],
                    "ROC_AUC": metrics["roc_auc"],
                    "PR_AUC": metrics["pr_auc"],
                })
            except Exception as e:
                print(f"  Error training {model_name}: {str(e)}")
    except Exception as e:
        print(f"  Error applying feature selection: {str(e)}")

results_df = pd.DataFrame(all_results)

print("\n" + "=" * 80)
print("COMPREHENSIVE RESULTS TABLE")
print("=" * 80)
if not results_df.empty:
    print(results_df.to_string(index=False))
else:
    print("No successful runs recorded. Check logs above.")

if not results_df.empty:
    results_df.to_csv("feature_selection_all_classifiers_results.csv", index=False)
    print("\n[INFO] Results saved to 'feature_selection_all_classifiers_results.csv'")

if not results_df.empty:
    print("\n" + "=" * 80)
    print("SUMMARY: AVERAGE PERFORMANCE BY FEATURE SELECTION METHOD")
    print("=" * 80)
    summary = results_df.groupby("Feature_Selection").agg({
        "N_Features": "first",
        "FS_Time_s": "first",
        "F1_Score": ["mean", "std"],
        "ROC_AUC": ["mean", "std"],
        "Accuracy": ["mean", "std"],
    }).round(4)
    summary.columns = [
        "N_Features",
        "FS_Time_s",
        "F1_Mean",
        "F1_Std",
        "ROC_AUC_Mean",
        "ROC_AUC_Std",
        "Acc_Mean",
        "Acc_Std",
    ]
    summary = summary.sort_values("F1_Mean", ascending=False)
    print(summary.to_string())
    print("\n" + "=" * 80)
    print("SUMMARY: AVERAGE PERFORMANCE BY MODEL")
    print("=" * 80)
    summary_model = results_df.groupby("Model").agg({
        "F1_Score": ["mean", "std"],
        "ROC_AUC": ["mean", "std"],
        "Accuracy": ["mean", "std"],
    }).round(4)
    summary_model.columns = [
        "F1_Mean",
        "F1_Std",
        "ROC_AUC_Mean",
        "ROC_AUC_Std",
        "Acc_Mean",
        "Acc_Std",
    ]
    summary_model = summary_model.sort_values("F1_Mean", ascending=False)
    print(summary_model.to_string())
    print("\n" + "=" * 80)
    print("BEST FEATURE SELECTION METHOD FOR EACH MODEL (by F1 Score)")
    print("=" * 80)
    for model_name in results_df["Model"].unique():
        model_results = results_df[results_df["Model"] == model_name]
        best_row = model_results.loc[model_results["F1_Score"].idxmax()]
        print(f"\n{model_name}:")
        print(f"  Best Method: {best_row['Feature_Selection']}")
        print(f"  N_Features: {best_row['N_Features']}")
        print(f"  F1 Score:   {best_row['F1_Score']:.4f}")
        print(f"  ROC-AUC:    {best_row['ROC_AUC']:.4f}")
        print(f"  Accuracy:   {best_row['Accuracy']:.4f}")
    print("\n" + "=" * 80)
    print("TOP 10 CONFIGURATIONS (by F1 Score)")
    print("=" * 80)
    top_10 = results_df.nlargest(10, "F1_Score")[
        ["Feature_Selection", "Model", "N_Features", "F1_Score", "ROC_AUC", "Accuracy"]
    ]
    print(top_10.to_string(index=False))
    best_global_row = results_df.loc[results_df["F1_Score"].idxmax()]
    print("\n" + "=" * 80)
    print("BEST OVERALL FEATURE SELECTION + MODEL CONFIGURATION")
    print("=" * 80)
    print(f"Feature Selection: {best_global_row['Feature_Selection']}")
    print(f"Model: {best_global_row['Model']}")
    print(f"N_Features: {best_global_row['N_Features']}")
    print(f"F1 Score: {best_global_row['F1_Score']:.4f}")
    print(f"ROC-AUC: {best_global_row['ROC_AUC']:.4f}")
    print(f"Accuracy: {best_global_row['Accuracy']:.4f}")

print("\n" + "=" * 80)
print("EXPERIMENT COMPLETE")
print("=" * 80)