# import shap
# import pandas as pd
# import joblib
# from data_loading import X_train


# # load trained model
# model = joblib.load("models/xgb_model_tuned.pkl")

# # create explainer
# explainer = shap.TreeExplainer(model)

# # compute SHAP values (use subset for speed)
# sample = X_train.sample(2000)

# shap_values = explainer.shap_values(sample)

# # calculate mean absolute importance
# importance = abs(shap_values).mean(axis=(0,2))

# feature_importance = pd.DataFrame({
#     "feature": X_train.columns,
#     "importance": importance
# })

# feature_importance = feature_importance.sort_values(
#     by="importance",
#     ascending=False
# )

# print(feature_importance)



import time
import numpy as np
from xgboost import XGBClassifier
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, classification_report
from data_loading import X_train, y_train, X_test, y_test, X_val, y_val
import joblib
import pandas as pd
import os

top15=[
"init_win_bytes_forward",
"init_win_bytes_backward",
"min_seg_size_forward",
"flow_iat_min",
"fwd_iat_min",
"bwd_packet_length_mean",
"bwd_packet_length_max",
"flow_iat_max",
"fwd_packet_length_max",
"fwd_iat_mean",
"flow_duration",
"fwd_iat_std",
"flow_iat_mean",
"flow_packets_per_s",
"flow_bytes_per_s"
]

# param_dist = {
#     # ── Tree count & learning rate (trade-off pair) ──────────
#     "n_estimators":   [100, 150, 200, 250],   # fewer trees = faster + smaller
#     "learning_rate":  [0.05, 0.08, 0.1, 0.15],

#     # ── Tree complexity ──────────────────────────────────────
#     "max_depth":        [4, 5, 6],            # shallower = smaller & faster
#     "min_child_weight": [3, 5, 7],            # higher = more regularisation

#     # ── Subsampling (speed + regularisation) ─────────────────
#     "subsample":        [0.7, 0.8, 0.9],
#     "colsample_bytree": [0.7, 0.8, 0.9],
#     "colsample_bylevel":[0.7, 0.8, 1.0],     # extra column sampling per level

#     # ── Regularisation (model size + accuracy) ───────────────
#     "gamma":      [0.0, 0.1, 0.2, 0.3],      # min loss-split threshold
#     "reg_alpha":  [0.0, 0.01, 0.1, 0.5],     # L1 → sparsity, fewer leaves
#     "reg_lambda": [0.5, 1.0, 1.5, 2.0],      # L2 → weight shrinkage
# }

# # ============================================================
# # BASE ESTIMATOR  (fixed settings that should never change)
# # ============================================================
# base_xgb = XGBClassifier(
#     objective="multi:softprob",
#     num_class=len(np.unique(y_train)),
#     tree_method="hist",          # fastest CPU method
#     eval_metric="mlogloss",
#     early_stopping_rounds=20,    # stops adding trees when val loss plateaus
#     n_jobs=-1,
#     random_state=42,
#     verbosity=0,
# )

# # ============================================================
# # RANDOMIZED SEARCH  (faster than GridSearchCV at high accuracy)
# # ============================================================
# cv = StratifiedKFold(n_splits=2, shuffle=True, random_state=42)

# search = RandomizedSearchCV(
#     estimator=base_xgb,
#     param_distributions=param_dist,
#     n_iter=15,                   # ↑ for more thorough search
#     scoring="f1_macro",       
#     cv=cv,
#     refit=True,                  # refit best model on full X_train
#     n_jobs=1,
#     verbose=2,
#     random_state=42,
# )


# # ── Fit  (pass eval_set so early_stopping_rounds works) ─────
# print("=" * 60)
# print("Starting RandomizedSearchCV …")
# print("=" * 60)

# start_search = time.time()
# search.fit(
#     X_train[top15], y_train,
#     eval_set=[(X_val[top15], y_val)],   # early stopping watches validation loss
# )
# end_search = time.time()

# print(f"\nSearch completed in {end_search - start_search:.1f}s")
# print(f"Best params:\n{search.best_params_}")
# print(f"Best CV macro F1: {search.best_score_:.4f}")


# ============================================================
# BEST MODEL  – retrain with best params & early stopping
# ============================================================
# best_params = search.best_params_

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
    subsample = 0.9, 
    reg_lambda=0.5,
    reg_alpha=0.1, 
    n_estimators=150, 
    min_child_weight=3, 
    max_depth=5, 
    learning_rate= 0.08, 
    gamma = 0.0, 
    colsample_bytree= 0.7, 
    colsample_bylevel=1.0
)

print("\n" + "=" * 60)
print("Retraining best model on full training set …")
print("=" * 60)

start_train = time.time()
best_xgb.fit(
    X_train[top15], y_train,
    eval_set=[(X_val[top15], y_val)],
    verbose=False,
)
end_train = time.time()

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

y_pred_best, t_best, acc_best, mf1_best, wf1_best = evaluate(
    best_xgb, X_val[top15], y_val, "TUNED MODEL (best params)"
)

# Test set
print("\n" + "=" * 60)
print("FINAL TEST-SET EVALUATION")
print("=" * 60)
y_pred_test, t_test, acc_test, mf1_test, wf1_test = evaluate(
    best_xgb, X_test[top15], y_test, "TUNED MODEL – Test Set"
)

if not os.path.exists("models/xgb_model_f15.pkl"):
    joblib.dump(best_xgb, "models/xgb_model_f15.pkl")
    print("Saved tuned model → xgb_model_f15.pkl")

results = pd.DataFrame({
    "Model":         ["XGb"],
    "Training time": [end_train - start_train],
    "Testing time":  [t_best],
    "Accuracy":      [acc_best],
    "Macro_f1":      [mf1_best],
    "F1 score":   [wf1_best]})

print(results.to_string(index=False))
 

if not os.path.exists("results/xgb_result_f15.csv"):
    results.to_csv("results/xgb_result_f15.csv", index=False) 



features=pd.Series(X_train[top15].columns)
if not os.path.exists("models/features_f15.pkl"):
    joblib.dump(features,"models/features_f15.pkl")


if not os.path.exists("deployment/X_test-top15.csv"): 
    X_test[top15].to_csv("deployment/X_test-top15.csv", index=False)   