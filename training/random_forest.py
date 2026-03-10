import time
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report
from training.data_loading import X_train, y_train, X_test, y_test,X_val, y_val
import joblib
import pandas as pd
import os

# -----------------------------
# Model Initialization
# -----------------------------
rf_model = RandomForestClassifier(
    n_estimators=200,        # good balance for 16GB RAM
    max_depth=None,          # allow full growth
    min_samples_split=2,
    min_samples_leaf=1,
    max_features='sqrt',     # standard for classification
    n_jobs=-1,               # use all CPU cores (M2)
    random_state=42
)

# -----------------------------
# Training
# -----------------------------

start_train = time.time()
rf_model.fit(X_train, y_train)   # DO NOT scale for RF
end_train = time.time()
if not os.path.exists("others/random_forest_model.pkl"):
    joblib.dump(rf_model, "others/random_forest_model.pkl")
# -----------------------------
# Testing
# -----------------------------

start_test = time.time()
y_pred = rf_model.predict(X_val)
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



result=pd.DataFrame({"Model":["Random forest"],
                     "Training time":[end_train - start_train],
                     "Testing time":[end_test - start_test],
                     "Accuracy":[accuracy],
                     "Macro_f1":[macro_f1],
                     "F1 score":[weighted_f1]})
if not os.path.exists("results/random_forest_result.csv"):
    result.to_csv("results/random_forest_result.csv")


# rf_model_loaded = joblib.load("random_forest_model.pkl")
# y_test_pred = rf_model_loaded.predict(X_test)
# accuracy_test = accuracy_score(y_test, y_test_pred)
# macro_f1_test = f1_score(y_test, y_test_pred, average='macro')
# weighted_f1_test = f1_score(y_test, y_test_pred, average='weighted')
# print("\nTest Set Classification Report:\n")
# print(classification_report(y_test, y_test_pred))
# print("Test Accuracy:", accuracy_test)
# print("Test Macro F1:", macro_f1_test)
# print("Test Weighted F1:", weighted_f1_test)    