import time
import numpy as np
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report
from data_loading import X_train, y_train, X_test, y_test, X_val, y_val
import joblib
import pandas as pd
import os

# -----------------------------
# Model Initialization
# -----------------------------
xgb_model = XGBClassifier(
    objective="multi:softprob",
    num_class=len(np.unique(y_train)),

    n_estimators=250,
    learning_rate=0.05,

    max_depth=6,
    min_child_weight=3,

    subsample=0.8,
    colsample_bytree=0.8,

    gamma=0.1,

    tree_method="hist",   # faster on CPU
    n_jobs=-1,
    random_state=42
)
# -----------------------------
# Training
# -----------------------------

start_train = time.time()
xgb_model.fit(X_train, y_train)   # DO NOT scale for XGB
end_train = time.time()
if not os.path.exists("xgb_model.pkl"):
    joblib.dump(xgb_model, "xgb_model.pkl")
# -----------------------------
# Testing
# -----------------------------

start_test = time.time()
y_pred = xgb_model.predict(X_val)
end_test = time.time()

# -----------------------------
# Metrics
# -----------------------------
accuracy = accuracy_score(y_val, y_pred)
macro_f1 = f1_score(y_val, y_pred, average='macro')
weighted_f1 = f1_score(y_val, y_pred, average='weighted')

print("Training Time:", end_train - start_train)
print("Testing Time:", end_test - start_test)
print("Accuracy:", accuracy)
print("Macro F1:", macro_f1)
print("Weighted F1:", weighted_f1)

print("\nClassification Report:\n")
print(classification_report(y_val, y_pred))


result=pd.DataFrame({"Model":["XGBoost"],
                     "Training time":[end_train - start_train],
                     "Testing time":[end_test - start_test],
                     "Accuracy":[accuracy],
                     "Macro_f1":[macro_f1],
                     "F1 score":[weighted_f1]})
if not os.path.exists("xgb_result.csv"):
    result.to_csv("xgb_result.csv")


# # -----------------------------
# # Test Set Evaluation
# # -----------------------------¯

# xg_model_loaded = joblib.load("xgb_model.pkl")
# y_test_pred = xg_model_loaded.predict(X_test)
# accuracy_test = accuracy_score(y_test, y_test_pred)
# macro_f1_test = f1_score(y_test, y_test_pred, average='macro')
# weighted_f1_test = f1_score(y_test, y_test_pred, average='weighted')
# print("\nTest Set Classification Report:\n")
# print(classification_report(y_test, y_test_pred))
# print("Test Accuracy:", accuracy_test)
# print("Test Macro F1:", macro_f1_test)
# print("Test Weighted F1:", weighted_f1_test)
