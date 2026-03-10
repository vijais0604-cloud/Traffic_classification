import time
import numpy as np
from xgboost import XGBClassifier
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, classification_report
from data_loading import X_train, y_train, X_test, y_test, X_val, y_val
import joblib
import pandas as pd
import os

# ============================================================
# HYPERPARAMETER SEARCH SPACE
# Goal: Improve speed & reduce model size without losing accuracy
#
# Key levers:
#   - Fewer/shallower trees  → faster inference, smaller model
#   - Higher learning_rate   → fewer trees needed (with early stopping)
#   - Lower max_depth        → smaller model, less overfitting
#   - Higher min_child_weight/ gamma → stronger regularisation
#   - reg_alpha / reg_lambda → L1/L2 pruning (shrinks model size)
# ============================================================

param_dist = {
    # ── Tree count & learning rate (trade-off pair) ──────────
    "n_estimators":   [100, 150, 200, 250],   # fewer trees = faster + smaller
    "learning_rate":  [0.05, 0.08, 0.1, 0.15],

    # ── Tree complexity ──────────────────────────────────────
    "max_depth":        [4, 5, 6],            # shallower = smaller & faster
    "min_child_weight": [3, 5, 7],            # higher = more regularisation

    # ── Subsampling (speed + regularisation) ─────────────────
    "subsample":        [0.7, 0.8, 0.9],
    "colsample_bytree": [0.7, 0.8, 0.9],
    "colsample_bylevel":[0.7, 0.8, 1.0],     # extra column sampling per level

    # ── Regularisation (model size + accuracy) ───────────────
    "gamma":      [0.0, 0.1, 0.2, 0.3],      # min loss-split threshold
    "reg_alpha":  [0.0, 0.01, 0.1, 0.5],     # L1 → sparsity, fewer leaves
    "reg_lambda": [0.5, 1.0, 1.5, 2.0],      # L2 → weight shrinkage
}

# ============================================================
# BASE ESTIMATOR  (fixed settings that should never change)
# ============================================================
base_xgb = XGBClassifier(
    objective="multi:softprob",
    num_class=len(np.unique(y_train)),
    tree_method="hist",          # fastest CPU method
    eval_metric="mlogloss",
    early_stopping_rounds=20,    # stops adding trees when val loss plateaus
    n_jobs=-1,
    random_state=42,
    verbosity=0,
)

# ============================================================
# RANDOMIZED SEARCH  (faster than GridSearchCV at high accuracy)
# ============================================================
cv = StratifiedKFold(n_splits=2, shuffle=True, random_state=42)

search = RandomizedSearchCV(
    estimator=base_xgb,
    param_distributions=param_dist,
    n_iter=15,                   # ↑ for more thorough search
    scoring="f1_weighted",       # optimise for weighted F1 (mirrors your goal)
    cv=cv,
    refit=True,                  # refit best model on full X_train
    n_jobs=1,
    verbose=2,
    random_state=42,
)

# ── Fit  (pass eval_set so early_stopping_rounds works) ─────
print("=" * 60)
print("Starting RandomizedSearchCV …")
print("=" * 60)

start_search = time.time()
search.fit(
    X_train, y_train,
    eval_set=[(X_val, y_val)],   # early stopping watches validation loss
)
end_search = time.time()

print(f"\nSearch completed in {end_search - start_search:.1f}s")
print(f"Best params:\n{search.best_params_}")
print(f"Best CV weighted-F1: {search.best_score_:.4f}")

# ============================================================
# BEST MODEL  – retrain with best params & early stopping
# ============================================================
best_params = search.best_params_

# Rebuild explicitly so we can control early_stopping_rounds
best_xgb = XGBClassifier(
    objective="multi:softprob",
    num_class=len(np.unique(y_train)),
    tree_method="hist",
    eval_metric="mlogloss",
    early_stopping_rounds=20,
    n_jobs=-1,
    random_state=42,
    verbosity=0,
    **best_params,
)

print("\n" + "=" * 60)
print("Retraining best model on full training set …")
print("=" * 60)

start_train = time.time()
best_xgb.fit(
    X_train, y_train,
    eval_set=[(X_val, y_val)],
    verbose=False,
)
end_train = time.time()

# ============================================================
# BASELINE MODEL  (original hyperparameters)
# ============================================================
baseline_xgb = XGBClassifier(
    objective="multi:softprob",
    num_class=len(np.unique(y_train)),
    n_estimators=250,
    learning_rate=0.05,
    max_depth=6,
    min_child_weight=3,
    subsample=0.8,
    colsample_bytree=0.8,
    gamma=0.1,
    tree_method="hist",
    n_jobs=-1,
    random_state=42,
    verbosity=0,
)

start_base_train = time.time()
baseline_xgb.fit(X_train, y_train)
end_base_train = time.time()

# ============================================================
# EVALUATION
# ============================================================
def evaluate(model, X, y, label="Model"):
    start = time.time()
    y_pred = model.predict(X)
    elapsed = time.time() - start

    acc  = accuracy_score(y, y_pred)
    mf1  = f1_score(y, y_pred, average="macro")
    wf1  = f1_score(y, y_pred, average="weighted")

    print(f"\n{'─'*50}")
    print(f"  {label}")
    print(f"{'─'*50}")
    print(f"  Inference time : {elapsed:.4f}s")
    print(f"  Accuracy       : {acc:.4f}")
    print(f"  Macro F1       : {mf1:.4f}")
    print(f"  Weighted F1    : {wf1:.4f}")
    print(f"\n{classification_report(y, y_pred)}")
    return y_pred, elapsed, acc, mf1, wf1

# Validation set
y_pred_base, t_base, acc_base, mf1_base, wf1_base = evaluate(
    baseline_xgb, X_val, y_val, "BASELINE (original params)"
)
y_pred_best, t_best, acc_best, mf1_best, wf1_best = evaluate(
    best_xgb, X_val, y_val, "TUNED MODEL (best params)"
)

# Test set
print("\n" + "=" * 60)
print("FINAL TEST-SET EVALUATION")
print("=" * 60)
y_pred_test, t_test, acc_test, mf1_test, wf1_test = evaluate(
    best_xgb, X_test, y_test, "TUNED MODEL – Test Set"
)

# ============================================================
# MODEL SIZE COMPARISON
# ============================================================
def model_size_kb(model, path):
    joblib.dump(model, path)
    size = os.path.getsize(path) / 1024
    return size

size_base = model_size_kb(baseline_xgb, "/tmp/xgb_baseline.pkl")
size_best = model_size_kb(best_xgb,     "/tmp/xgb_best.pkl")

print(f"\n{'='*60}")
print(f"  Model size – Baseline : {size_base:.1f} KB")
print(f"  Model size – Tuned    : {size_best:.1f} KB  "
      f"({'smaller' if size_best < size_base else 'larger'} by "
      f"{abs(size_best - size_base):.1f} KB)")
print(f"{'='*60}\n")

# ============================================================
# SAVE BEST MODEL
# ============================================================
if not os.path.exists("xgb_model_tuned.pkl"):
    joblib.dump(best_xgb, "xgb_model_tuned.pkl")
    print("Saved tuned model → xgb_model_tuned.pkl")

# ============================================================
# RESULTS CSV
# ============================================================
results = pd.DataFrame({
    "Model":         ["XGBoost Baseline", "XGBoost Tuned"],
    "Training time": [end_base_train - start_base_train, end_train - start_train],
    "Testing time":  [t_base, t_best],
    "Accuracy":      [acc_base, acc_best],
    "Macro_f1":      [mf1_base, mf1_best],
    "F1 score":   [wf1_base, wf1_best]})

print(results.to_string(index=False))

if not os.path.exists("xgb_result_tuned.csv"):
    results.to_csv("xgb_result_tuned.csv", index=False)
    print("\nSaved results → xgb_result_tuned.csv")