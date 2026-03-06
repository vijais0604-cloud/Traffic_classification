import time
import numpy as np
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report
from data_loading import X_train, y_train, X_test, y_test, X_val, y_val
import joblib
import pandas as pd
import matplotlib.pyplot as plt

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

# ------------------------------
# Model Performance Visualization
# -------------------------------
result=pd.DataFrame({"Model":["XGBoost"],
                     "Training time":[end_train - start_train],
                     "Testing time":[end_test - start_test],
                     "Accuracy":[accuracy],
                     "Macro_f1":[macro_f1],
                     "F1 score":[weighted_f1]})

fig , axs = plt.subplots(3,1)
axs[0].bar(result["Model"],result["Accuracy"],color="blue",label="Accuracy")
axs[1].bar(result["Model"],result["Macro_f1"],color="orange",label="Macro F1")
axs[2].bar(result["Model"],result["F1 score"],color="green",label="Weighted F1")
axs.set_title("Model Performance Metrics")
axs.set_ylabel("Score")
axs.legend()
plt.ylim(0, 1)
plt.tight_layout
plt.show()  



# # -----------------------------
# # Test Set Evaluation
# # -----------------------------

# rf_model_loaded = joblib.load("xgb_model.pkl")
# y_test_pred = rf_model_loaded.predict(X_test)
# accuracy_test = accuracy_score(y_test, y_test_pred)
# macro_f1_test = f1_score(y_test, y_test_pred, average='macro')
# weighted_f1_test = f1_score(y_test, y_test_pred, average='weighted')
# print("\nTest Set Classification Report:\n")
# print(classification_report(y_test, y_test_pred))
# print("Test Accuracy:", accuracy_test)
# print("Test Macro F1:", macro_f1_test)
# print("Test Weighted F1:", weighted_f1_test)
