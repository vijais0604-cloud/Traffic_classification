import time
import joblib
from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report
import os
from training.data_loading import X_train, y_train, X_val, y_val
import pandas as pd

# Model
lgb_model = LGBMClassifier(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=-1,
    class_weight='balanced',
    random_state=42
)

# Training
start_train = time.time()
lgb_model.fit(X_train, y_train)
end_train = time.time()

if not os.path.exists("others/lightgbm_model.pkl"):
    joblib.dump(lgb_model, "others/lightgbm_model.pkl")

# Testing
start_test = time.time()
y_pred = lgb_model.predict(X_val)
end_test = time.time()

# Metrics
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

result=pd.DataFrame({"Model":["lightGBM"],
                     "Training time":[end_train - start_train],
                     "Testing time":[end_test - start_test],
                     "Accuracy":[accuracy],
                     "Macro_f1":[macro_f1],
                     "F1 score":[weighted_f1]})
if not os.path.exists("results/lgb_result.csv"):
    result.to_csv("results/lgb_result.csv")


# # -----------------------------
# # Test Set Evaluation
# # -----------------------------¯

# lg_model_loaded = joblib.load("lightgbm_model.pkl")
# y_test_pred = lg_model_loaded.predict(X_test)
# accuracy_test = accuracy_score(y_test, y_test_pred)
# macro_f1_test = f1_score(y_test, y_test_pred, average='macro')
# weighted_f1_test = f1_score(y_test, y_test_pred, average='weighted')
# print("\nTest Set Classification Report:\n")
# print(classification_report(y_test, y_test_pred))
# print("Test Accuracy:", accuracy_test)
# print("Test Macro F1:", macro_f1_test)
# print("Test Weighted F1:", weighted_f1_test)
